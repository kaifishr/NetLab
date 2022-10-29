"""Collection of custom layers for neural networks.

"""
import torch
from torch import nn
from torch.nn import functional as F


class Linear(nn.Module):
    """Linear layer to simulation evolution from simple to complex.

    Typical usage example:

        def evolve_layer(prob=prob):
            for module in model.modules():
                if isinstance(module, Linear):
                    setattr(module, "prob", prob)
                    module.mask()

        prob = 0.01
        evolve_layer(prob=prob)

        for epoch in range(epochs):

            for x, y in trainloader():
                ...

            prob += 0.01
            evolve_layer(prob=prob)

    """
    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = torch.nn.Parameter(torch.empty(out_features))

        self.prob = None 
        self.rand =torch.nn.Parameter(torch.rand_like(input=self.weight), requires_grad=False)
        self.gate =torch.nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)
            
        torch.nn.init.normal_(self.weight, mean=0.0, std=0.02)
        torch.nn.init.zeros_(self.bias)

    @torch.no_grad()
    def mask(self):
        self.gate.data = torch.where(self.rand < self.prob, 1.0, 0.0)

    # @torch.no_grad()
    # def mask(self):
    #     rand =torch.rand_like(input=self.weight)    # remove self.rand in __init__
    #     self.gate.data += torch.where(rand < self.prob, 1.0, 0.0)
    #     self.gate.data = torch.where(self.gate > 0.0, 1.0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.gate * self.weight, self.bias)


class Conv2d(nn.Conv2d):
    """Layer to simulation evolution from simple to complex."""

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, padding: str
    ):

        super(Conv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.prob = None 
        self.rand =torch.nn.Parameter(torch.rand_like(input=self.weight), requires_grad=False)
        self.gate =torch.nn.Parameter(torch.zeros_like(self.weight), requires_grad=False)

    @torch.no_grad()
    def mask(self):
        self.gate.data = torch.where(self.rand < self.prob, 1.0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input=x,
            weight=self.gate * self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )
