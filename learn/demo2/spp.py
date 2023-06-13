import torch
import torch.nn as nn
import torch.nn.functional as F

class SPP(nn.Module):
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPP, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        # 获取输入张量的形状
        num, c, h, w = x.size()

        # 定义 SPP 层的输出张量列表
        output = []

        # 对于每个池化层级别，计算池化层大小和步幅
        for i in range(self.num_levels):
            level = i + 1
            size = (h // level, w // level)
            stride = (h // level, w // level)

            # 根据池化类型进行池化操作
            if self.pool_type == 'max_pool':
                pool = F.max_pool2d(x, kernel_size=size, stride=stride).view(num, -1)
            else:
                pool = F.avg_pool2d(x, kernel_size=size, stride=stride).view(num, -1)

            # 将池化结果添加到输出列表中
            output.append(pool)

        # 将每个池化层的结果拼接在一起
        output = torch.cat(output, dim=1)

        return output