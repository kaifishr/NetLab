"""Collection of custom layers for neural networks.

"""
import torch
from torch import nn


class Linear(nn.Module):
    """Model to simulation evolution from simple to complex."""

    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features)))
        self.bias = torch.nn.Parameter(torch.empty(out_features))

        self.prob = torch.nn.Parameter(torch.tensor(0.0, requires_grad=False))
        self.rand = torch.nn.Parameter(
            torch.rand_like(input=self.weight), requires_grad=False
        )
        self.gate = torch.nn.Parameter(
            torch.zeros_like(self.weight), requires_grad=False
        )

    @torch.no_grad()
    def mask(self):
        self.gate.data = torch.where(
            self.rand < self.prob,
            torch.ones_like(self.weight),
            torch.zeros_like(self.weight),
        )

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

        self.prob = torch.nn.Parameter(torch.tensor(0.0, requires_grad=False))
        self.rand = torch.nn.Parameter(
            torch.rand_like(input=self.weight), requires_grad=False
        )
        self.gate = torch.nn.Parameter(
            torch.zeros_like(self.weight), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(
            input=x,
            weight=self.gate * self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    @torch.no_grad()
    def mask(self):
        self.gate.data = torch.where(
            self.rand < self.prob,
            torch.ones_like(self.weight),
            torch.zeros_like(self.weight),
        )
