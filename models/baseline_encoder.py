import torch
import torch.nn as nn
import torch.nn.functional as F
from models.self_attention import Self_Attn


class Encoder(nn.Module):
    def __init__(self, out_dim=64, base_model='encoder'):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.att = Self_Attn(64, 'relu')

        # projection MLP
        self.l1 = nn.Linear(64, 64)
        self.l2 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = F.relu(x)
        h = self.pool(x)

        # get the attention of the representation
        obj_main, obj_bg, attention = self.att(h)

        return self.project(obj_main), self.project(obj_bg)  # , obj_main, obj_bg, attention

    def project(self, x):
        x1 = torch.mean(x, dim=[2, 3])
        x1 = self.l1(x1)
        x1 = F.relu(x1)
        x1 = self.l2(x1)

        return x1
