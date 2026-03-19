import torch
import torch.optim as optim
from tqdm import tqdm
import time
import utils.tools as tools
from utils import data_loader
import models.MSJ as MSJ
import random
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
pic_path = "./result/2_train/"
dataname = 'segc390'
name = "MSJ"+dataname
os.makedirs(pic_path,exist_ok=True)
train_dataloader,valid_dataloader = data_loader.get_dataloader(
    '../datasets/3d_datas/'+dataname+'_train.npy')
net = MSJ.MSJ(feat_channels=[16,32,64,128,256],rotate_4=False)
model0 = net.cuda()

optimizer = optim.Adam(model0.parameters(), lr=0.001)
# training
step = 0
start = datetime.now()
print('Start training.....', start.strftime("%H:%M:%S"))

# loss_list = []
# loss_valid_list = []
# snr_valid_list = []

best_val_loss = float('inf')
epochs_no_improve_after_lr = 0
best_model_state = None

factor = 0.5
patience_lr = 5
patience_es = 10
min_lr = 1e-6

scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=factor,
        patience=patience_lr,
        min_lr=min_lr
    )

for epoch in tqdm(range(200)):
    loss_epoch = 0

    optimizer.zero_grad()
    # train list
    tmp_loss = 0
    j = 0
    for patch,mask in train_dataloader:
        patch = patch[0]
        mask = mask[0]
        patch_size = patch.shape
        zero = 0.5
        patch = patch.cuda()
        
        # 生成添加了二次掩码的数据，0则不添加
        random_number = 0
        
        # 添加额外的伪mask
        mask_i = tools.random_zero_matrix_3d(mask[:,:,0], random_number)

        mask = mask.cuda()

        mask_i = torch.from_numpy(mask_i).int().cuda()

        mask_diff = torch.logical_xor((mask_i).cuda(),mask).int().cuda()

        data_masked_fake = (patch * mask_i + zero*(1-mask_i)).reshape(1, 1, *patch_size)

        data_label = (patch).reshape(1,1,*patch_size)

        # 生成输入与label
        input_train = data_masked_fake.float().cuda()
        label_train = data_label.float().cuda()

        # training step
        model0.train()
        out_train_input = model0(input_train)
        out_train = out_train_input[0, 0]


        # loss1 = tools.total_loss(out_train * (1-mask_i_gpu), label_train * (1-mask_i_gpu))
        loss1 = tools.total_loss(out_train * mask_i, label_train * mask_i)
        # loss2 = tools.total_loss(out_train *mask_diff, label_train * mask_diff)

        loss_first = loss1
        loss_first.backward()
        if (j+1)%4==0:
            optimizer.step()
            optimizer.zero_grad()
        j+=1
        loss_epoch += loss_first.item()
    loss_epoch/=len(train_dataloader.dataset)
    
    # validation
    val_loss = 0
    val_snr = 0
    model0.eval()
    with torch.no_grad():
        for patch,mask in valid_dataloader:
            patch = patch[0].reshape(1, 1, *patch_size).float().cuda()
            valid_out = model0(patch)

            mask = mask[0].cuda()
            loss = tools.total_loss(valid_out[0, 0]*mask, patch[0, 0]*mask)
            snr,mse,psnr = tools.snr_get(valid_out[0, 0]*mask, patch[0, 0]*mask)
            val_loss += loss.item()
            val_snr+=snr
       
        val_loss /= len(valid_dataloader.dataset)
        val_snr /= len(valid_dataloader.dataset)
        print(f"Epoch {epoch:03d}: Train Loss: {loss_epoch:.6f}, Val Loss: {val_loss:.4f}, Val SNR: {val_snr:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model0.state_dict()
        epochs_no_improve_after_lr = 0
    else:
        epochs_no_improve_after_lr += 1

    # 更新学习率调度器
    scheduler.step(val_loss)
    current_lrs = scheduler.get_last_lr()
    print(f"Current learning rates: {current_lrs}")

    # 检查学习率是否降低过
    current_lrs = [group['lr'] for group in optimizer.param_groups]
    if any(lr <= min_lr + 1e-12 for lr in current_lrs):
        # 如果已到最小学习率，并且验证损失连续 patiencelr + patience_es 轮无改进，则提前停止
        if epochs_no_improve_after_lr >= patience_es:
            print(f"Early stopping at epoch {epoch} (no improvement after LR reduction)")
            break

if best_model_state is not None:
        model0.load_state_dict(best_model_state)

torch.save(model0, pic_path + str(epoch) + name + time.strftime('%Y-%m-%d-%H', time.localtime()) + '.pt')
