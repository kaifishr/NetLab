"""Common modules for neural networks."""
import torch
import torch.nn as nn

from src.config import Config
from src.modules.layer import DenseLayer


class PatchEmbedding(nn.Module):
    """Per-patch linear embedding."""

    def __init__(self, config: Config) -> None:
        """Initializes PatchEmbedding module."""
        super().__init__()

        patch_size = config.densenet.patch_size
        num_dim_hidden = config.densenet.num_dim_hidden
        embedding_channels = config.densenet.embedding_channels

        img_channels, img_height, img_width = config.data.input_shape

        assert (img_height % patch_size == 0) and (img_width % patch_size == 0)

        dim_embedding = (
            embedding_channels * (img_height // patch_size) * (img_width // patch_size)
        )

        self.conv = nn.Conv2d(
            in_channels=img_channels,
            out_channels=embedding_channels,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )

        self.linear = nn.Linear(in_features=dim_embedding, out_features=num_dim_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(input=x, start_dim=1, end_dim=-1)
        x = self.linear(x)
        return x


class DenseBlock(nn.Module):
    """Dense block."""

    def __init__(
        self, in_features: int, out_features: int, num_hidden: int = 2
    ) -> None:
        super().__init__()

        blocks = []
        for _ in range(num_hidden):
            blocks += [DenseLayer(in_features=in_features, out_features=out_features)]

        self.dense_block = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dense_block(x)


class ConvBlock(nn.Module):
    def __init__(self, num_channels: int):
        super().__init__()

        conv_cfg = dict(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=3,
            padding="same",
        )

        self.conv1 = nn.Conv2d(**conv_cfg)
        self.conv2 = nn.Conv2d(**conv_cfg)

        self.gelu = nn.GELU()

        self.bn1 = torch.nn.BatchNorm2d(num_features=num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_features=num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        identity = x

        out = self.conv1(x)
        out = self.gelu(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = self.gelu(out)
        out = self.bn2(out)

        out += identity

        return out
