import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # print(output1.size(), output2.size(), len(label))
        # exit()
        distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean((1 - label) * torch.pow(distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))


        return loss
