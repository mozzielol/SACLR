import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Split_KL_sum(nn.Module):
    def __init__(self, ntx_loss, get_softmax=False):
        super(Split_KL_sum, self).__init__()
        self.get_softmax = get_softmax
        self.ntx_loss = ntx_loss
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, global_feature, splits):
        loss = 0.0
        for att, outputs in splits:
            summation = torch.flatten(outputs * att, start_dim=1)
            if self.get_softmax:
                summation = F.softmax(summation, dim=1)
                global_feature = F.softmax(global_feature, dim=1)
            M = ((global_feature + summation) / 2).log()
            # loss += (self.kl(M, global_feature) + self.kl(M, summation)) / 2
            loss += self.ntx_loss(summation, global_feature)
        return loss


class Split_KL(nn.Module):
    def __init__(self, ntx_loss, get_softmax=True):
        super(Split_KL, self).__init__()
        self.get_softmax = get_softmax
        self.ntx_loss = ntx_loss
        self.kl = nn.KLDivLoss(reduction='batchmean')

    def forward(self, global_feature, splits):
        loss = 0.0
        for att, outputs in splits:
            attention = torch.sum(att, dim=1).unsqueeze(1)
            W, H = attention.size(2), attention.size(3)
            for i in np.arange(0, W):
                for j in np.arange(0, H):
                    local_feature = outputs[:, :, i, j]
                    local_att = attention[:, :, i, j]
                    if self.get_softmax:
                        local_feature = F.softmax(local_feature, dim=1)
                        global_feature = F.softmax(global_feature, dim=1)
                    M = ((global_feature + local_feature) * local_att / 2).log()
                    # loss += (self.kl(M, global_feature * local_att) + self.kl(M, local_feature * local_att)) / 2
                    loss += self.ntx_loss(local_feature, global_feature) * torch.mean(local_att) / (W * H)
        return loss
