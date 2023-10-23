import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class DirectNormLoss(nn.Module):
    def __init__(self, num_class=81, nd_weight=1.0):
        super(DirectNormLoss, self).__init__()
        self.num_class = num_class
        self.nd_weight = nd_weight

    def project_center(self, s_emb, t_emb, T_EMB, labels):
        assert s_emb.size() == t_emb.size()
        assert s_emb.shape[0] == len(labels)
        loss = 0.0
        for s, t, i in zip(s_emb, t_emb, labels):
            i = i.item()
            center = torch.tensor(T_EMB[str(i)]).cuda()
            e_c = center / center.norm(p=2)
            max_norm = max(s.norm(p=2), t.norm(p=2))
            loss += 1 - torch.dot(s, e_c) / max_norm
        return loss

    def forward(self, s_emb, t_emb, T_EMB, labels):
        nd_loss = self.project_center(s_emb=s_emb, t_emb=t_emb, T_EMB=T_EMB, labels=labels) * self.nd_weight

        return nd_loss / len(labels)


class KDLoss(nn.Module):
    '''
	Distilling the Knowledge in a Neural Network
	https://arxiv.org/pdf/1503.02531.pdf
	'''

    def __init__(self, kd_weight=1.0, T=1.0):
        super(KDLoss, self).__init__()
        self.T = T
        self.kd_weight = kd_weight

    def forward(self, s_out, t_out):
        kd_loss = F.kl_div(F.log_softmax(s_out / self.T, dim=1),
                           F.softmax(t_out / self.T, dim=1),
                           reduction='batchmean',
                           ) * self.T * self.T
        return kd_loss * self.kd_weight
