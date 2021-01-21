import torch.nn as nn
import torch
import torch.nn.functional as F


class Sim_cluster(nn.Module):
    def __init__(self, in_features, num_cluster=10):
        super(Sim_cluster, self).__init__()
        self.projector = nn.Linear(in_features, num_cluster, bias=False)
        self.tao = 0.001
        self.num_cluster = num_cluster
        self.criterion = nn.BCELoss(reduction='sum')

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        similarity = self.projector(x)
        scores = F.softmax(similarity / self.tao)
        return self.compute_loss(similarity, scores)

    def compute_loss(self, similarity, scores):
        '''
        :param similarity: cos similarity of the inputs to the code (N * C)
        :param scores: softmax of the similarity (N * C)
        :return: loss (N * 1)
        '''
        loss = 0
        for idx in range(self.num_cluster):
            indices = scores[:, idx]
            probs = torch.sum(torch.exp(similarity[:, idx]) * indices) / \
                                   torch.sum(torch.exp(similarity[:, idx]))
            loss += self.criterion(probs, torch.ones_like(probs))
        return loss
