import torch.nn as nn
import torch.nn.functional as F

from learn.demo2.model.spp import SPP as SPP

# resnet50 + SPP
class Bottleneck(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(Bottleneck, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel),
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(Bottleneck, 64, 3, stride=1)
        self.layer2 = self.make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self.make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self.make_layer(Bottleneck, 512, 3, stride=2)
        self.spp = SPP([2])
        self.fc = nn.Linear(2048,32)#-->（batch,32）

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    # -->（batch,3,32,32）
    def forward_once(self, x):
        out = self.conv1(x)     #-->（batch,64,32,32）
        out = self.layer1(out)  #-->（batch,64,32,32）
        out = self.layer2(out)  #-->（batch,128,16,16）
        out = self.layer3(out)  #-->（batch,256,8,8）
        out = self.layer4(out)  #-->（batch,512,4,4）
        # out = F.avg_pool2d(out, 4)     #-->（batch,512,1,1）
        out = self.spp(out)
        # out = out.view(out.size(0), -1) #-->（batch,512*1*1）
        out = self.fc(out)      #-->（batch,32）
        # out = F.log_softmax(out, dim=1)
        return out

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
