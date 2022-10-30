"""Collection of custom neural networks.

"""
import torch
import torch.nn as nn

from src.config import Config
from .module import PatchEmbedding, ConvBlock, DenseBlock


class ConvNet(nn.Module):
    """Isotropic convolutional neural network with residual connections."""

    def __init__(self, config: Config):
        super().__init__()

        self.input_shape = config.data.input_shape
        self.n_channels_in = self.input_shape[0]
        self.n_dims_out = config.data.num_classes
        self.n_channels_hidden = config.convnet.n_channels_hidden
        self.n_channels_out = config.convnet.n_channels_out
        self.n_blocks = config.convnet.n_blocks

        self.features = self._feature_extractor()
        self.classifier = nn.Linear(
            self.n_channels_out * (self.input_shape[-1] // 4) ** 2, self.n_dims_out
        )

        self.apply(self._weights_init)

    def _weights_init(self, module: nn.Module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

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
            nn.GELU(),
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
        self.num_dim_out = config.data.num_classes
        self.num_dim_hidden = config.densenet.num_dim_hidden
        self.num_blocks = config.densenet.num_blocks
        self.num_hidden = config.densenet.num_hidden

        self.patch_embedding = PatchEmbedding(config=config)
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
        blocks = []

        for _ in range(self.num_blocks):
            blocks.append(
                DenseBlock(
                    in_features=self.num_dim_hidden,
                    out_features=self.num_dim_hidden,
                    num_hidden=self.num_hidden,
                )
            )

        # Output layer
        blocks.append(nn.Linear(self.num_dim_hidden, self.num_dim_out))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.classifier(x)
        return x
