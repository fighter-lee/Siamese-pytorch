from torch import nn
from torch.nn import L1Loss
import torch

input = torch.tensor([1,2,3])
target = torch.tensor([1,2,5])

input = torch.reshape(input, (1,1,1,3))
target = torch.reshape(target, (1,1,1,3))

print(input)

loss = L1Loss(reduction='sum')
result = loss(input, target)

print(result)

x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x, (1,3))
loss = nn.CrossEntropyLoss()
result_cross = loss(x, y)
print(result_cross)