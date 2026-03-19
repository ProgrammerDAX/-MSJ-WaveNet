import numpy as np

import random

import matplotlib.pyplot as plt

import torch

import math

def total_loss(fixed,label):

    beta = 0.5*label.max()
    diff = torch.abs(fixed-label)
    log_cosh = torch.log(torch.cosh(fixed-label))
    cond = diff < beta
    loss = torch.where(cond, 0.5 * diff ** 2,beta* (diff - 0.5 * beta))
    loss = (torch.mean(loss+log_cosh))
    # loss = torch.mean(torch.abs(fixed-label))
    return loss


def snr_get(out, label):
    """
    输入: out, label — 可以是 NumPy 数组 或 PyTorch 张量（CPU 或 CUDA）
    返回: snr, mse, psnr （Python float 值）
    """
    # 如果是 PyTorch Tensor，先 detach grad，再转 CPU，再转 NumPy
    if isinstance(out, torch.Tensor):
        out_np = out.detach().cpu().numpy()
    else:
        out_np = np.array(out)

    if isinstance(label, torch.Tensor):
        label_np = label.detach().cpu().numpy()
    else:
        label_np = np.array(label)

    # 确保形状一致
    assert out_np.shape == label_np.shape, "out 和 label 必须形状一致"

    # 计算
    square = np.sum(np.square(label_np))
    l2 = np.sum(np.square(out_np - label_np))
    mse = l2 / (np.prod(out_np.shape))
    psnr = 10 * np.log10((np.max(label_np)**2) / mse)
    snr = 10 * np.log10(square / l2)

    return float(snr), float(mse), float(psnr)

def random_zero_matrix_3d(matrix, p):
    # 将输入的列表转换为NumPy数组以便于操作
    matrix = np.array(matrix)
    
    # 生成一个与matrix形状相同的随机矩阵
    random_mask = np.random.random(matrix.shape)
    
    # 在二维矩阵上应用概率操作：将1以概率p变为0
    matrix[(matrix == 1) & (random_mask < p)] = 0
    
    # 将二维矩阵延展到第三维，深度为128
    matrix_3d = matrix[:, :, np.newaxis]  # 增加一个维度
    matrix_3d = np.tile(matrix_3d, (1, 1, 128))  # 沿第三维复制128次
    
    return matrix_3d
