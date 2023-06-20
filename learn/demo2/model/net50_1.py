import torch.nn as nn
import torch.nn.functional as F

from learn.demo2.model.resnet import resnet50
from learn.demo2.model.spp import SPP

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.inchannel = 64
        self.resnet = resnet50(include_top=False)
        self.spp = SPP([2])
        self.fc = nn.Linear(8192, 32)#-->（batch,32）

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    # -->（batch,3,32,32）
    def forward_once(self, x):
        out = self.resnet(x)
        # out = F.avg_pool2d(out, 4)     #-->（batch,512,1,1）
        out = self.spp(out)
        # out = out.view(out.size(0), -1) #-->（batch,512*1*1）
        out = self.fc(out)      #-->（batch,32）
        out = F.log_softmax(out, dim=1)
        return out

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
