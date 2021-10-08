'''
MemVir
Copyright (c) 2021-present NAVER Corp.
Apache License v2.0
'''
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Embedding, self).__init__()
        self.embedding = nn.Linear(input_dim, output_dim)

    def forward(self, input_tensor, l2_norm=True):
        x = self.embedding(input_tensor)

        if l2_norm:
            x = F.normalize(x, p=2, dim=1)

        return x


def embedding(input_dim=1024, output_dim=64):
    module = Embedding(input_dim=input_dim, output_dim=output_dim)

    return module
