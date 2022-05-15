"""Script holds tools for network manipulation."""
import os
import random
import numpy as np

from textwrap import wrap

import torch
from torch import nn


def init_weights(module: torch.nn.Module) -> None:
    """Initializes weights in network.

    Example:
        model = resnet18()
        model.apply(init_weights)

    Args:
        module:

    Returns:

    """
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        torch.nn.init.kaiming_uniform_(module.weight.data, nonlinearity="linear")
        # torch.nn.init.xavier_uniform_(module.weight.data, gain=1)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias.data)


def replace_module(module, old_module, new_module) -> None:
    """Replaces modules in existing PyTorch model.

    Example:

        # Replace activation function
        replace_module(module=model, old_module=torch.nn.ReLU, new_module=torch.nn.GELU)

    Args:
        module: PyTorch model.
        old_module: Module that is to be replaced.
        new_module: Module replacing old module.

    """
    for name, child in module.named_children():
        if isinstance(child, old_module):
            setattr(module, name, new_module())
        else:
            replace_module(child, old_module, new_module)


def save_checkpoint(model: nn.Module, ckpt_dir: str, model_name: str) -> None:
    """Save model checkpoint.

    Args:
        model:
        ckpt_dir:
        model_name:

    """
    model_path = os.path.join(ckpt_dir, f"{model_name}.pth")
    torch.save(obj=model.state_dict(), f=model_path)


def load_checkpoint(model: nn.Module, ckpt_dir: str, model_name: str) -> None:
    """Load model from checkpoint.

    Args:
        model:
        ckpt_dir:
        model_name:

    """
    model_path = os.path.join(ckpt_dir, f"{model_name}.pth")
    state_dict = torch.load(f=model_path)
    model.load_state_dict(state_dict=state_dict)


def set_attribute(model: torch.nn.Module, attribute: str, value) -> None:
    """

    Example:

        # Set dropout rate manually:
        set_attribute(model=model, attribute="dropout_rate", value=0.5)


    Args:
        model:
        attribute:
        value:

    """
    for module in model.modules():
        if hasattr(module, attribute):
            setattr(module, attribute, value)


def count_model_parameters(model: nn.Module,
                           is_trainable: bool = True,
                           verbose: bool = True) -> int:
    """Counts model parameters.

    Args:
        model: PyTorch model.
        is_trainable: Count only trainable parameters if true.
        verbose: Print number of trainable parameters.

    Returns:
        Number of model parameters.

    """
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad is is_trainable)

    if verbose:
        print(f"Number of trainable parameters: {'.'.join(wrap(str(n_params)[::-1], 3))[::-1]}.")

    return n_params


def set_random_seed(seed: int = 69) -> None:
    """Controls sources of randomness.

    This method is not bulletproof.
    See also: https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        seed: Random seed.

    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
