import time

import matplotlib
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

def getAccuracy(output1, output2, label):
    distance = F.pairwise_distance(output1, output2)
    zero_labels = label == 0  # 找出labels中为0的位置
    predict_for_zero_labels = distance[zero_labels.squeeze()]  # 找出predict中对应位置的浮点类型数
    less_than_1_15 = predict_for_zero_labels < 1.15  # 找出浮点类型数小于1.15的位置
    accuracy = less_than_1_15.sum().item()

    one_labels = label == 1  # 找出labels中为1的位置
    predict_for_one_labels = distance[one_labels.squeeze()]  # 找出predict中对应位置的浮点类型数
    greater_than_1_15 = predict_for_one_labels > 1.15  # 找出浮点类型数大于1.15的位置
    accuracy += greater_than_1_15.sum().item()  # 计算浮点类型数大于1.15的数量
    return accuracy

def showImgArray(img_1, img_2, label):
    matplotlib.use('TkAgg')
    img1 = torch.from_numpy(img_1.numpy())
    img2 = torch.from_numpy(img_2.numpy())

    # 创建一个3x8的子图网格
    fig, ax = plt.subplots(nrows=3, ncols=8, figsize=(24, 9))
    gs = plt.GridSpec(3, 8, figure=fig, height_ratios=[1, 1, 0.1])
    # 将图像和标签添加到子图网格中
    for i in range(8):
        col = i % 8
        ax0 = plt.subplot(gs[0, col])
        ax0.imshow(img1[i].numpy().transpose((1, 2, 0)))
        ax0.axis("off")
        ax1 = plt.subplot(gs[1, col])
        ax1.imshow(img2[i].numpy().transpose((1, 2, 0)))
        ax1.axis("off")
        ax2 = plt.subplot(gs[2, col])
        ax2.text(0.5, 0.5, f"label={label[i][0]}", ha="center", va="center", fontsize=12)
        ax2.axis("off")

    plt.tight_layout()

    # 将子图网格保存到文件
    plt.savefig("image_grid.png")