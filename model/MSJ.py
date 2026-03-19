
import torch
import torch.nn as nn
from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool3d,GroupNorm
from torch.nn import ReLU, Sigmoid
import torch.nn as nn
import torch
from .utils import common
# 创建时间：2023.01.01 dax


class Conv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
                        common.pad_cat(),
                        Conv3d(inp_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True, padding_mode='circular'),
                        
                        BatchNorm3d(out_feat),
                        ReLU()
                        # common.CustomLeakyReLU()
                        )

        self.conv2 = Sequential(
                        common.pad_cat(),
                        Conv3d(out_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True, padding_mode='circular'),
                        
                        BatchNorm3d(out_feat),
                        ReLU()
                        # common.CustomLeakyReLU()
                        )
        self.conv3 = Sequential(
                        common.pad_cat(),
                        Conv3d(out_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True, padding_mode='circular'),
                        
                        BatchNorm3d(out_feat),
                        ReLU()
                        # common.CustomLeakyReLU()
                        )

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Sequential(
                common.pad_cat(),
                Conv3d(inp_feat, out_feat, kernel_size=1, bias=False, padding_mode='circular')
            )

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv3(self.conv2(self.conv1(x))) + self.residual_upsampler(res)



class outconv(nn.Module):

    def __init__(self, in_ch, out_ch,rotate_4):
        super(outconv, self).__init__()
        if rotate_4:
            self.conv = nn.Sequential(
            common.pad_cat(),
            # rotate_back(),
            common.rotate_back4(),
            nn.Conv3d(in_ch*4, in_ch*2, 1, padding_mode='circular'),#这里本来都是*2，前面现在改成*4了
            nn.BatchNorm3d(in_ch*2),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_ch*2, out_ch, 1, padding_mode='circular'),
            
            # common.CustomLeakyReLU()
            ReLU()
        )
        else:
            self.conv = nn.Sequential(
            common.pad_cat(),
            common.rotate_back(),
            nn.Conv3d(in_ch*2, in_ch*2, 1, padding_mode='circular'),
            nn.BatchNorm3d(in_ch*2),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_ch*2, out_ch, 1, padding_mode='circular'),
            
            Sigmoid()
            
        )
        

    def forward(self, x):
        
        x = self.conv(x)

        return x

class MSJ(Module):
    # __                            __
    #  1|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    #           4|__ __ __|4
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

    def __init__(self,num_channels=1,feat_channels=[16,32,64,128,256], residual='conv',rotate_4 = False):

        #residual: 是否加入残差边，不加则为None
        super(MSJ, self).__init__()

        # DWT & IWT
        self.DWT = common.DWT()
        self.IWT = common.IWT()

        self.conv_DWT1 = Conv3D_Block(feat_channels[0]*8, feat_channels[0], residual=residual)
        self.conv_DWT2 = Conv3D_Block(feat_channels[1]*8, feat_channels[1], residual=residual)
        self.conv_DWT3 = Conv3D_Block(feat_channels[2]*8, feat_channels[2], residual=residual)
        self.conv_DWT4 = Conv3D_Block(feat_channels[3]*8, feat_channels[3], residual=residual)


        # self.rotate = rotate()
        if rotate_4:
            self.rotate = common.rotate4()
        else:
            self.rotate = common.rotate()
        # Encoder convolutions

        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4]*4, residual=residual)

        # Decoder convolutions
        self.dec_conv_blk4 = Conv3D_Block(2*feat_channels[3], 4*feat_channels[3], residual=residual)
        self.dec_conv_blk3 = Conv3D_Block(2*feat_channels[2], 4*feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2*feat_channels[1], 4*feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2*feat_channels[0], 2*feat_channels[0], residual=residual)


        # attentions
        self.attention4 = common.Attention_block(feat_channels[3])
        self.attention3 = common.Attention_block(feat_channels[2])
        self.attention2 = common.Attention_block(feat_channels[1])
        self.attention1 = common.Attention_block(feat_channels[0])
        

        self.outconv = outconv(2*feat_channels[0],num_channels,rotate_4)

        # Activation function
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # x (1,1,128,128,128)
        # 编码器

        # 新改进：将下采样的pool替换成了DWT
        x1 = self.rotate(x)# (2,1,128,128,128)

        x1 = self.conv_blk1(x1) #(2,16,128,128,128)
        x_low1 = self.DWT(x1) #(2,128,64,64,64)
        x_low1 = self.conv_DWT1(x_low1)#(2,16,128,64,64)

        x2 = self.conv_blk2(x_low1)#(2,32,128,64,64)
        x_low2 = self.DWT(x2)#(2,32*4,128,32,32)
        x_low2 = self.conv_DWT2(x_low2)#(2,32,128,32,32)

        x3 = self.conv_blk3(x_low2)#(2,64,128,32,32)
        
        x_low3 = self.DWT(x3)#(2,64*4,128,16,16)
        x_low3 = self.conv_DWT3(x_low3)#(2,64,128,16,16)

        x4 = self.conv_blk4(x_low3)#(2,128,128,16,16)
        x_low4 = self.DWT(x4)#(2,128*4,128,8,8)
        x_low4 = self.conv_DWT4(x_low4)#(2,128,128,8,8)

        base = self.conv_blk5(x_low4)#(2,128*8,128,8,8)
        base = self.IWT(base)#(2,128,128,16,16)

        # x4 = self.attention4(x4)
        # print(x4.device)
        # print(base.device)
        d4 = torch.cat([base, x4], dim=1)

        d_high4 = self.IWT(self.dec_conv_blk4(d4))
        # x3 = self.attention3(x3)
        d3 = torch.cat([d_high4, x3], dim=1)

        d_high3 = self.IWT(self.dec_conv_blk3(d3))

        # x2 = self.attention2(x2)
        d2 = torch.cat([(d_high3), x2], dim=1)

        d_high2 = self.IWT(self.dec_conv_blk2(d2))
        # x1 = self.attention1(x1)
        d1 = torch.cat([(d_high2), x1], dim=1)
        
        # d1 = self.CBAM4(d1)
        d_high1 = self.dec_conv_blk1(d1)
        
        seg = self.outconv(d_high1)

        return seg



if __name__=='__main__':

    net = MSJ().cuda(1)

    import torch
    x = torch.rand(1, 1, 128, 128, 128).cuda(1)
    out = net.forward(x)
    print(out.size())
    print(1)
