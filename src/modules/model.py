"""Collection of custom neural networks.

"""
from math import prod
import torch
import torch.nn as nn

from .block import ConvBlock, DenseBlock


class ConvNet(nn.Module):
    """Isotropic convolutional neural network with residual connections."""

    def __init__(self, config: dict):
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

        self._weights_init()

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
        for i in range(self.n_blocks):
            layers.append(ConvBlock(num_channels=self.n_channels_hidden))

        # Conv network out
        layers += [
            nn.Conv2d(
                in_channels=self.n_channels_hidden,
                out_channels=self.n_channels_out,
                kernel_size=2,
                stride=2,
            ),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=self.n_channels_out),
        ]

        return nn.Sequential(*layers)

    def _weights_init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data, nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = self.classifier(x)
        return x


class DenseNet(nn.Module):
    """Isotropic fully connected neural network with residual connections."""

    def __init__(self, config: dict):
        super().__init__()

        self.input_shape = config.data.input_shape
        self.n_dims_in = prod(self.input_shape)
        self.n_dims_out = config.data.n_classes
        self.n_dims_hidden = config.densenet.n_dims_hidden
        self.n_blocks = config.densenet.n_blocks

        self.classifier = self._make_classifier()

        self._weights_init()

    def _make_classifier(self):
        layers = []

        # Input layer
        layers += [
            nn.Linear(in_features=self.n_dims_in, out_features=self.n_dims_hidden),
            nn.BatchNorm1d(num_features=self.n_dims_hidden),
        ]

        # Hidden layer
        for i in range(self.n_blocks):
            layers.append(
                DenseBlock(
                    in_features=self.n_dims_hidden, out_features=self.n_dims_hidden
                )
            )

        # Output layer
        layers += [nn.Linear(self.n_dims_hidden, self.n_dims_out)]

        return nn.Sequential(*layers)

    def _weights_init(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight.data, nonlinearity="relu")
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias.data)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x
