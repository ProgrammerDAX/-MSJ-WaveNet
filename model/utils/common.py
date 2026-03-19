import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class pad_cat(nn.Module):


    def __init__(self):
        super(pad_cat, self).__init__()

    def forward(self, x):
        x1 = torch.nn.functional.pad(x[:,:,:,:-1,:],pad=(0,0,1,0,0,0,0,0,0,0), mode="constant",value=0.5)
        x1[:,:,:,0,:] = x[:,:,:,2,:]
        return x1
#20250401 改动：旋转方向改为第一个batch不变，第二个沿着time 轴水平旋转180度，平移特征张量改为对对倒数第二维度平移

class rotate(nn.Module):

    def __init__(self):
        super(rotate, self).__init__()

    def forward(self, x):
        # x1 = torch.rot90(x, k=2, dims=[2,3])
        x1 = torch.flip(x, dims=[2,3])

        # x2 = torch.rot90(x, k=-1, dims=[3,4])

        # x2 = torch.rot90(x2,k=2, dims=[2,3])

        x = torch.cat((x,x1),0)
        return x
    

class pad_cat_old(nn.Module):


    def __init__(self):
        super(pad_cat_old, self).__init__()

    def forward(self, x):
        x1 = torch.nn.functional.pad(x[:,:,:,:,:-1],pad=(1,0,0,0,0,0,0,0,0,0), mode="constant",value=0.5)
        x1[:,:,:,:,0] = x[:,:,:,:,1]
        return x1
#20250401 改动：旋转方向改为第一个batch不变，第二个沿着time 轴水平旋转180度，平移特征张量改为对对倒数第二维度平移

class rotate_old(nn.Module):

    def __init__(self):
        super(rotate_old, self).__init__()

    def forward(self, x):
        x1 = torch.rot90(x, k=1, dims=[3,4])
        # x1 = torch.flip(x, dims=[2,3])

        # x2 = torch.rot90(x, k=-1, dims=[3,4])

        # x2 = torch.rot90(x2,k=2, dims=[2,3])

        x = torch.cat((x,x1),0)
        return x

class rotate4(nn.Module):

    def __init__(self):
        super(rotate4, self).__init__()

    def forward(self, x):
        x1 = torch.rot90(x, k=1, dims=[2,3])
        
        x2 = torch.rot90(x, k=2, dims=[2,3])
        
        x3 = torch.rot90(x, k=3, dims=[2,3])

        # x2 = torch.rot90(x2,k=2, dims=[2,3])

        x = torch.cat((x,x1,x2,x3),0)
        return x
    
#旋转改动了，那么反转也要跟着改，注意是逆向操作

class rotate_back(nn.Module):


    def __init__(self):
        super(rotate_back, self).__init__()
        

    def forward(self, x):
        f,s,h,w,t=x.shape
        batch_size = f//2
        
        x1 = x[:batch_size]
        # x2 = torch.rot90(x[batch_size:2*batch_size,:,:,:,:], k=2, dims=[2,3])
        x2 = torch.flip(x[batch_size:2*batch_size,:,:,:,:], dims=[2,3])
        
        # x2 = torch.rot90(x[batch_size:2*batch_size,:,:,:,:], k=1, dims=[3,4])

        x = torch.cat((x1,x2),1)
        return x
    
class rotate_back_old(nn.Module):


    def __init__(self):
        super(rotate_back_old, self).__init__()
        

    def forward(self, x):
        f,s,h,w,t=x.shape
        batch_size = f//2
        
        x1 = x[:batch_size]
        x2 = torch.rot90(x[batch_size:2*batch_size,:,:,:,:], k=-1, dims=[3,4])
        # x2 = torch.flip(x[batch_size:2*batch_size,:,:,:,:], dims=[2,3])
        

        x = torch.cat((x1,x2),1)
        return x
    
class rotate_back4(nn.Module):


    def __init__(self):
        super(rotate_back4, self).__init__()
        

    def forward(self, x):
        f,s,h,w,t=x.shape
        batch_size = f//4
        x1 = x[:batch_size,:,:,:,:]
        
        x2 = torch.rot90(x[batch_size:2*batch_size,:,:,:,:], k=-1, dims=[2,3])

        x3 = torch.rot90(x[2*batch_size:3*batch_size,:,:,:,:], k=2, dims=[2,3])

        x4 = torch.rot90(x[3*batch_size:4*batch_size,:,:,:,:], k=-3, dims=[2,3])

        x = torch.cat((x1,x2,x3,x4),1)
        return x


