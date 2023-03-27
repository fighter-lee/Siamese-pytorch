import torch
import torch.nn as nn

from nets.ResNet import resnet50
from nets.vgg import VGG16


def get_img_output_length(width, height):
    def get_output_length(input_length):
        # input_length += 6
        filter_sizes = [2, 2, 2, 2, 2]
        padding = [0, 0, 0, 0, 0]
        stride = 2
        for i in range(5):
            input_length = (input_length + 2 * padding[i] - filter_sizes[i]) // stride + 1
        return input_length
    return get_output_length(width) * get_output_length(height) 
    
class Siamese(nn.Module):
    def __init__(self, input_shape, pretrained=False, useResnet=True):
        super(Siamese, self).__init__()
        self.useResnet = useResnet
        if useResnet:
            self.resnet = resnet50(pretrained, include_top=True)
            del self.resnet.avgpool
            del self.resnet.fc
            # flat_shape = 2048 * get_img_output_length(input_shape[1], input_shape[0])
            flat_shape = 2048 * 3 * 3
            # flat_shape = 512 * 4 * 4
        else:
            self.vgg = VGG16(pretrained, 3)
            del self.vgg.avgpool
            del self.vgg.classifier
            flat_shape = 512 * get_img_output_length(input_shape[1], input_shape[0])
        self.fully_connect1 = torch.nn.Linear(flat_shape, 512)
        self.fully_connect2 = torch.nn.Linear(512, 1)

    def forward(self, x):
        x1, x2 = x
        if self.useResnet:
            # ------------------------------------------#
            #   我们将两个输入传入到主干特征提取网络
            # ------------------------------------------#
            x1 = self.resnet.conv1(x1)
            x1 = self.resnet.bn1(x1)
            x1 = self.resnet.relu(x1)
            x1 = self.resnet.maxpool(x1)

            x1 = self.resnet.layer1(x1)
            x1 = self.resnet.layer2(x1)
            x1 = self.resnet.layer3(x1)
            x1 = self.resnet.layer4(x1)

            x2 = self.resnet.conv1(x2)
            x2 = self.resnet.bn1(x2)
            x2 = self.resnet.relu(x2)
            x2 = self.resnet.maxpool(x2)

            x2 = self.resnet.layer1(x2)
            x2 = self.resnet.layer2(x2)
            x2 = self.resnet.layer3(x2)
            x2 = self.resnet.layer4(x2)
            # -------------------------#
            #   相减取绝对值
            # -------------------------#
            x1 = torch.flatten(x1, 1)
            x2 = torch.flatten(x2, 1)
            x = torch.abs(x1 - x2)
            # -------------------------#
            #   进行两次全连接
            # -------------------------#
            x = self.fully_connect1(x)
            x = self.fully_connect2(x)
        else:
            #------------------------------------------#
            #   我们将两个输入传入到主干特征提取网络
            #------------------------------------------#
            x1 = self.vgg.features(x1)
            x2 = self.vgg.features(x2)
            #-------------------------#
            #   相减取绝对值，取l1距离
            #-------------------------#
            x1 = torch.flatten(x1, 1)
            x2 = torch.flatten(x2, 1)
            x = torch.abs(x1 - x2)
            #-------------------------#
            #   进行两次全连接
            #-------------------------#
            x = self.fully_connect1(x)
            x = self.fully_connect2(x)
        return x
