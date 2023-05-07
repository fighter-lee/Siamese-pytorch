
import torchvision
from torch import nn

# torchvision.datasets.ImageNet("data_image_net", split='train', download=True, transform=torchvision.transforms.ToTensor())

dataset = torchvision.datasets.CIFAR10("data", train=False, transform=torchvision.transforms.ToTensor(),  download=True)

vgg16 = torchvision.models.vgg16(pretrained=True)

print(vgg16)

# 使用原有的网络结构，增加一层线性层
vgg16.classifier.add_module("add_linear", nn.Linear(1000, 10))

print(vgg16)

# 不使用默认权重，直接修改vgg16的结构
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print("vgg16_false", vgg16_false)