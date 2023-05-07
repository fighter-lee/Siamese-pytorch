import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Sigmoid
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Model1(nn.Module):

    def __init__(self):
        super(Model1, self).__init__()
        # 非线性变换，让模型泛化能力更强
        self.sigmoid1 = Sigmoid()

    def forward(self, x):
        x = self.sigmoid1(x)
        return x

model1 = Model1()

writer = SummaryWriter("relu_logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = model1(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)
    writer.add_images('output', output, step)
    step += 1

# 查看经过卷积后的图像：tensorboard --logdir=logs