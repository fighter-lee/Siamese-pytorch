import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SPP(nn.Module):
    def __init__(self, levels=[1, 2, 4]):
        super(SPP, self).__init__()
        self.levels = levels

    def forward(self, x):
        bs, c, h, w = x.shape
        x = x.view(bs, c, h, w)

        for i in range(len(self.levels)):
            level = self.levels[i]
            bins_h = torch.arange(0, h + 1, h // level)
            bins_w = torch.arange(0, w + 1, w // level)

            if i == 0:
                spp = F.max_pool2d(x, kernel_size=(bins_h[1] - bins_h[0], bins_w[1] - bins_w[0]),
                                   stride=(bins_h[1] - bins_h[0], bins_w[1] - bins_w[0]))
                spp = spp.view(bs, -1)
            else:
                for h in range(level):
                    for w in range(level):
                        h_start, h_end = bins_h[h], bins_h[h + 1]
                        w_start, w_end = bins_w[w], bins_w[w + 1]
                        roi = x[:, :, h_start:h_end, w_start:w_end]
                        max_pool = F.max_pool2d(roi, kernel_size=(h_end - h_start, w_end - w_start),
                                                stride=(h_end - h_start, w_end - w_start))
                        spp = torch.cat((spp, max_pool.view(bs, -1)), dim=1)

        return spp