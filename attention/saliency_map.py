import torch.nn as nn
import torch.nn.functional as F
import torch


class Saliency_vgg(nn.Module):
    """ Self attention Layer"""

    def __init__(self):
        super(Saliency_vgg, self).__init__()
        self.conv_1 = nn.Conv2d(512, 512, 3)  # 512 * 12 * 12
        self.conv_2 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.conv_3 = nn.Conv2d(512, 512, 3, padding=(1, 1))  # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool2d((2, 2), padding=(1, 1))  # pooling 512 * 7 * 7

    def forward(self, x):
        out = self.conv_1(x)  # 12
        out = F.relu(out)
        out = self.conv_2(out)  # 12
        out = F.relu(out)
        out = self.conv_3(out)  # 12
        out = F.relu(out)
        # out = self.maxpool5(out)  # 7
        out = F.sigmoid(out)

        return out

