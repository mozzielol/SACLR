import torch.nn as nn
import torch.nn.functional as F
import torch
from util.device import get_device


class Siam(nn.Module):
    def __init__(self, batch_size):
        super(Siam, self).__init__()
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss(reduction="sum").to(get_device())

    def forward(self, p, z):
        # p = F.normalize(p, p=2, dim=1)
        # z = F.normalize(z, p=2, dim=1)
        labels = torch.zeros(self.batch_size).to(get_device()).long()
        return self.criterion(self.gaussian_activation((p - z).pow(2).sum(dim=1, keepdim=True)), labels)
    
    def gaussian_activation(self, x, sigma=1.):
        return torch.exp(-x / (2 * sigma * sigma))