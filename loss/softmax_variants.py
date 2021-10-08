'''
MemVir
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import init


class NormSoftmax(nn.Module):
    def __init__(self, input_dim, n_classes, scale=1.0, memvir=None):
        super(NormSoftmax, self).__init__()
        self.scale = scale
        self.n_classes = n_classes

        self.fc_weight = Parameter(torch.Tensor(input_dim, n_classes))
        init.kaiming_uniform_(self.fc_weight, a=math.sqrt(5))

        # For Proxy Memory
        self.memvir = memvir

    def forward(self, input, target):
        # input is already l2_normalized

        # Use MemVir
        if self.memvir is not None:
            fc_weight = self.memvir.prepare_proxy(self.fc_weight)
            input, target = self.memvir.prepare_emb(input, target)
        else:
            fc_weight = self.fc_weight

        fc_weight_l2 = F.normalize(fc_weight, p=2, dim=0)
        
        logits = input.matmul(fc_weight_l2)
        batch_size = target.size(0)

        # Get gt logits
        pos_logits = logits[torch.arange(0, batch_size), target].view(-1, 1)
        target = target.view(-1, 1)

        # positive terms
        logits.scatter_(1, target.data, pos_logits)
        logits = self.scale * logits

        loss = F.cross_entropy(logits, target.view(-1))

        return loss


class ProxyNCA(nn.Module):
    def __init__(self, input_dim, n_classes, scale=3.0, init_type='normal', memvir=None):
        super(ProxyNCA, self).__init__()
        self.fc_weight = Parameter(torch.Tensor(input_dim, n_classes))
        self.n_classes = n_classes
        self.scale = scale
        if init_type == 'normal':
            init.kaiming_normal_(self.fc_weight, mode='fan_out')
        elif init_type == 'uniform':
            init.kaiming_uniform_(self.fc_weight, a=math.sqrt(5))
        else:
            raise ValueError('%s not supported' % init_type)
        
        self.memvir = memvir

    def forward(self, input, target):
        # Use MemVir
        if self.memvir is not None:
            fc_weight = self.memvir.prepare_proxy(self.fc_weight)
            input, target = self.memvir.prepare_emb(input, target)
        else:
            fc_weight = self.fc_weight
        
        # input already l2_normalized
        proxy_l2 = F.normalize(fc_weight, p=2, dim=0)

        # N, dim, cls
        dist_mat = torch.cdist(input, proxy_l2.t()) ** 2
        dist_mat *= self.scale
        
        _, new_n_classes = fc_weight.shape
        pos_target = F.one_hot(target, new_n_classes).float()
           
        loss = torch.mean(torch.sum(-pos_target * F.log_softmax(-dist_mat, -1), -1)) # with positive

        return loss
