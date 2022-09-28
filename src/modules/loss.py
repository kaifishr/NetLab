"""Custom loss functions.

"""
import torch
import torch.nn as nn


class EntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        p = self.softmax(self.flatten(x))
        h = -1.0 * (p * torch.log(torch.clamp(p, min=1e-5))).mean()
        return h
