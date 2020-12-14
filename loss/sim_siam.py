import torch.nn as nn
import torch.nn.functional as F


class Siam(nn.Module):
    def __init__(self):
        super(Siam, self).__init__()

    def forward(self, p, z):
        z = z.detach()
        p = F.normalize(p, p=2, dim=1)
        z = F.normalize(z, p=2, dim=1)
        return - (p * z).sum(dim=1).mean()
