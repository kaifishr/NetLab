"""Common modules for neural networks."""
import random

import torch
import torch.nn as nn

from src.config import Config


class AveragePool(nn.Module):

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.mean(x, dim=self.dim)
        return x


class PatchEmbedding(nn.Module):
    """Per-patch linear embedding."""

    def __init__(self, config: Config) -> None:
        """Initializes PatchEmbedding module."""
        super().__init__()

        img_channels, img_height, img_width = config.data.input_shape
        patch_size = config.module.patch_embedding.patch_size

        assert (img_height % patch_size == 0) and (img_width % patch_size == 0)

        model_dim = config.module.mlp_mixer.model_dim

        self.conv = nn.Conv2d(
            in_channels=img_channels,
            out_channels=model_dim,
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(input=x, start_dim=2, end_dim=-1)
        x = torch.swapaxes(x, axis0=-2, axis1=-1)
        return x


class MlpBlock(nn.Module):

    def __init__(self, dim: int, config: Config):
        super().__init__()

        hidden_dim = config.module.mlp_block.hidden_dim

        self.mlp_block = nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=dim),
        )

    def forward(self, x: torch.Tensor):
        x = self.mlp_block(x)
        return x

class SwapAxes(nn.Module):

    def __init__(self, axis0: int, axis1):
        super().__init__()
        self.axis0 = axis0
        self.axis1 = axis1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.swapaxes(x, axis0=self.axis0, axis1=self.axis1)


class MixerBlock(nn.Module):

    def __init__(self, config: Config):
        super().__init__()

        model_dim = config.module.mlp_mixer.model_dim

        _, img_height, img_width = config.data.input_shape
        patch_size = config.module.patch_embedding.patch_size
        patch_dim = (img_height // patch_size) * (img_width // patch_size)

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(model_dim),
            SwapAxes(axis0=-2, axis1=-1),
            MlpBlock(patch_dim, config),
            SwapAxes(axis0=-2, axis1=-1),
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(model_dim),
            MlpBlock(model_dim, config)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.token_mixer(x)
        x = x + self.channel_mixer(x)
        return x


class DenseBlock(nn.Module):
    """Dense block."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()

        num_hidden = 4
        layers = []

        for _ in range(num_hidden):
            layers += [
                nn.Linear(in_features=in_features, out_features=out_features),
                nn.LayerNorm(out_features),
                nn.Dropout(p=0.1),
                nn.ReLU(),
            ]
        self.mlp = nn.Sequential(*layers)
        
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mlp(x)
        return x
