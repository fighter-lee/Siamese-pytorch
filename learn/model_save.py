import torch
import torchvision

vgg16 = torchvision.models.vgg16(False)

# 保存模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 保存 方式2，只会保存模型参数
torch.save(vgg16.state_dict(), "vgg16_method2.pth")

model = torch.load("vgg16_method1.pth")
print(model)

vgg16_2 = torchvision.models.vgg16(False)
model2 = torch.load("vgg16_method2.pth")
vgg16_2.load_state_dict(model2)
print(vgg16_2)