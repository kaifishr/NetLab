import torch
import torch.nn as nn


class Sawtooth(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.b = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.m * x) + self.b


class ReLU2(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.b = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.a * x) - torch.relu(-self.b * x)


class ReLU3_(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.a = torch.nn.Parameter(torch.ones(size=(in_features,), requires_grad=True))
        self.b = torch.nn.Parameter(torch.ones(size=(in_features,), requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.a * x) - torch.relu(-self.b * x)


class ReLU3(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self.init_weights = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.init_weights:
            self.a = torch.nn.Parameter(torch.ones(size=x.shape[1:], device="cuda:0"), requires_grad=True)
            self.b = torch.nn.Parameter(torch.ones(size=x.shape[1:], device="cuda:0"), requires_grad=True)
            self.init_weights = False
        return torch.relu(self.a * x) - torch.relu(-self.b * x)


class Peak(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.a = torch.nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.b = torch.nn.Parameter(torch.tensor(0.0, requires_grad=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.0 / (1.0 + (self.a * x + self.b) ** 2)
