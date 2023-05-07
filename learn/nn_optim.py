import torch
import torchvision.datasets
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor(),  download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # self.conv1 = Conv2d(3, 32, 5, padding=2)
        # self.maxpool1 = MaxPool2d(2)
        # # 输入和输出的channel相等，需要padding=2
        # self.conv2 = Conv2d(32, 32, 5, padding=2)
        # self.maxpool2 = MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, 5, padding=2)
        # self.maxpool3 = MaxPool2d(2)
        # self.flatten = Flatten()
        # self.linear1 = Linear(1024, 64)
        # self.linear2 = Linear(64, 10)

        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10),
        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x


model = Model()
print(model)
# input = torch.ones((64, 3, 32, 32))
# print(input.shape)
# output = model(input)
# print(output.shape)

loss = nn.CrossEntropyLoss()

optim = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        # print(imgs)
        output = model(imgs)
        # print(output)
        # print(targets)
        result_loss = loss(output, targets)
        # print(result_loss)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        running_loss = running_loss + result_loss
    print(epoch, running_loss)
