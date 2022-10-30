"""Collection of custom layers for neural networks.

"""
import torch
from torch import nn

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
