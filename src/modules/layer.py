"""Collection of custom layers for neural networks.

"""
import torch
from torch import nn
from torch.nn import functional as F


class DenseLayer(nn.Module):
    """Dense Layer."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        self.dense_layer = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.GELU(),
            nn.LayerNorm(out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dense_layer(x)
        return x


class SimpleComplexLinear(nn.Module): 
    """Linear layer to simulation evolution from simple to complex.

    Typical usage:

        prob = 0.01
        evolve_layer(model=model, prob=prob)

        for inputs, labels in train_loader:

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

        prob += 0.001
        evolve_layer(model=model, prob=prob)  # tools.py
    
    """
    def __init__(self, in_features: int, out_features: int, bias: bool=True):
        super().__init__()

        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = torch.nn.Parameter(torch.empty(out_features)) if bias else None

        self.rand = torch.nn.Parameter(torch.rand_like(input=self.weight), requires_grad=False)
        self.gate = torch.nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
        self.prob = torch.nn.Parameter(torch.tensor(0.0), requires_grad=False)

    @property
    def probability(self) -> float:
        return self.prob.item()

    @probability.setter
    def probability(self, prob: float) -> None:
        self.prob.copy_(max(0.0, min(prob, 1.0)))

    @torch.no_grad()
    def evolve(self):
        self.gate.data = torch.where(self.rand < self.prob, 1.0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return F.linear(x, self.gate * self.weight, self.bias)
        if self.prob < 1.0:
            return F.linear(x, self.gate * self.weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)