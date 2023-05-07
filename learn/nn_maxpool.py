import torch
from torch import nn
from torch.nn import Conv2d, MaxPool2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Model1(nn.Module):

    def __init__(self):
        super(Model1, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, x):
        x = self.maxpool1(x)
        return x

model1 = Model1()

writer = SummaryWriter("maxpool_logs")

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