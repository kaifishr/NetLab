"""Collection of custom neural networks.

"""
import random
from math import prod
import torch
import torch.nn as nn

from src.config import Config
from .block import ConvBlock
from .module import PatchEmbedding, DenseBlock, MixerBlock, AveragePool


class MlpMixer(nn.Module):

    def __init__(self, config: Config) -> None:
        super().__init__()

        num_blocks = config.module.mlp_mixer.num_blocks
        model_dim = config.module.mlp_mixer.model_dim
        num_classes = config.data.num_classes

        self.patch_embedding = PatchEmbedding(config)

        mixer_blocks = [MixerBlock(config) for _ in range(num_blocks)]
        self.mixer = nn.Sequential(*mixer_blocks)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(model_dim),
            AveragePool(dim=-2),
            nn.Linear(in_features=model_dim, out_features=num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        x = self.patch_embedding(x)
        x = self.mixer(x)
        x = self.mlp_head(x)
        return x


class ConvNet(nn.Module):
    """Isotropic convolutional neural network with residual connections."""

    def __init__(self, config: Config):
        super().__init__()

        self.input_shape = config.data.input_shape
        self.n_channels_in = self.input_shape[0]
        self.n_dims_out = config.data.n_classes
        self.n_channels_hidden = config.convnet.n_channels_hidden
        self.n_channels_out = config.convnet.n_channels_out
        self.n_blocks = config.convnet.n_blocks

        self.features = self._feature_extractor()
        self.classifier = nn.Linear(
            self.n_channels_out * (self.input_shape[-1] // 4) ** 2, self.n_dims_out
        )

        self.apply(self._weights_init)

    def _weights_init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data, nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)

    def _feature_extractor(self):
        layers = []

        # Conv network input
        layers += [
            nn.Conv2d(
                in_channels=self.n_channels_in,
                out_channels=self.n_channels_hidden,
                kernel_size=2,
                stride=2,
            ),
            nn.BatchNorm2d(num_features=self.n_channels_hidden),
        ]

        # Conv network hidden
        for _ in range(self.n_blocks):
            layers.append(ConvBlock(num_channels=self.n_channels_hidden))

        # Conv network out
        layers += [
            nn.Conv2d(
                in_channels=self.n_channels_hidden,
                out_channels=self.n_channels_out,
                kernel_size=2,
                stride=2,
            ),
            nn.GeLU(),
            nn.BatchNorm2d(num_features=self.n_channels_out),
        ]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.classifier(x)
        return x


class DenseNet(nn.Module):
    """Isotropic fully connected neural network with residual connections."""

    def __init__(self, config: Config):
        super().__init__()

        self.input_shape = config.data.input_shape
        self.num_dim_in = prod(self.input_shape)
        self.num_dim_out = config.data.num_classes
        self.num_dim_hidden = config.densenet.num_dim_hidden
        self.num_blocks = config.densenet.num_blocks
        self.num_hidden = config.densenet.num_hidden

        self.classifier = self._make_classifier()

        self.apply(self._weights_init)

    def _weights_init(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)

    def _make_classifier(self):
        layers = []

        # Input layer
        layers += [
            nn.Linear(in_features=self.num_dim_in, out_features=self.num_dim_hidden),
            nn.LayerNorm(self.num_dim_hidden),
        ]

        # Hidden layer
        for _ in range(self.num_blocks):
            layers.append(
                DenseBlock(
                    in_features=self.num_dim_hidden, 
                    out_features=self.num_dim_hidden,
                    num_hidden=self.num_hidden
                )
            )

        # Output layer
        layers += [nn.Linear(self.num_dim_hidden, self.num_dim_out)]

        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
