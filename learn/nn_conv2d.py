import torch
from torch import nn
from torch.nn import Conv2d
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10('data', train=False, download=True, transform=torchvision.transforms.ToTensor())
dataloader = DataLoader(dataset, batch_size=64)

class Model1(nn.Module):

    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x

model1 = Model1()

writer = SummaryWriter("logs")

step = 0
for data in dataloader:
    imgs, targets = data
    output = model1(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)

    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images('output', output, step)
    step += 1

# 查看经过卷积后的图像：tensorboard --logdir=logs