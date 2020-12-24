import torch
import torch.nn as nn
from torch.nn import functional as F


class rSoftMax(nn.Module):
    def __init__(self, radix, cardinality, at_map):
        super().__init__()
        assert radix > 0
        self.radix = radix
        self.cardinality = cardinality
        self.at_map = at_map

    def forward(self, x):
        batch, ch, H, W = x.size()
        if self.at_map:
            x = x.view(batch, ch, -1)
            x = F.softmax(x, dim=-1)
            return x.view(batch, ch, H, W)
        if self.radix > 1:
            x = x.view(batch, self.cardinality, self.radix, -1).transpose(1, 2)
            x = F.softmax(x, dim=1)
            x = x.reshape(batch, -1)
        else:
            x = torch.sigmoid(x)
        return x


class Splat(nn.Module):
    def __init__(self, channels, radix, cardinality, reduction_factor=4, at_map=True):
        super(Splat, self).__init__()
        self.radix = radix
        self.cardinality = cardinality
        self.channels = channels
        self.at_map = at_map
        inter_channels = max(channels * radix // reduction_factor, 32)
        self.fc1 = nn.Conv2d(channels, inter_channels, 1, groups=cardinality)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.relu = nn.ReLU(inplace=True)
        if at_map:
            self.fc2 = nn.Conv2d(inter_channels, channels, 1, groups=cardinality)
        else:
            self.fc2 = nn.Conv2d(inter_channels, channels * radix, 1, groups=cardinality)
        self.rsoftmax = rSoftMax(radix, cardinality, at_map)

    def forward(self, x):
        batch, rchannel = x.shape[:2]
        if self.radix > 1:
            splited = torch.split(x, rchannel // self.radix, dim=1)
            gap = x
        else:
            gap = x
        if not self.at_map:
            gap = F.adaptive_avg_pool2d(gap, 1)
        gap = self.fc1(gap)

        gap = self.bn1(gap)
        gap = self.relu(gap)

        atten = self.fc2(gap)
        atten = self.rsoftmax(atten)
        if not self.at_map:
            atten = atten.view(batch, -1, 1, 1)

        if self.radix > 1:
            attens = torch.split(atten, rchannel // self.radix, dim=1)
            # return sum([att * split for (att, split) in zip(attens, splited)])
            return [(att, split) for (att, split) in zip(attens, splited)]
        else:
            out = atten * x
        return out.contiguous()
