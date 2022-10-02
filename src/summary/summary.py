"""
summary.py

Script holds methods for Tensorboard.
"""
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from ..config.config import Config


def add_graph(
    model: nn.Module, 
    dataloader: DataLoader, 
    writer: SummaryWriter, 
    config: Config
) -> None:
    """Add graph of model to Tensorboard.

    Args:
        model:
        dataloader:
        writer:
        config: Class holding configuration.
    """
    device = config.trainer.device
    x_data, _ = next(iter(dataloader))
    writer.add_graph(model=model, input_to_model=x_data.to(device))


def add_input_samples(
    dataloader: DataLoader,
    tag: str,
    writer: SummaryWriter,
    global_step: int = 0,
    n_samples: int = 16,
) -> None:
    """Add samples from dataloader to Tensorboard.

    Check if the input to the model is as expected.

    Args:
        dataloader:
        tag:
        writer:
        global_step:
        n_samples:

    """
    x, _ = next(iter(dataloader))
    n_samples = min(x.size(0), n_samples)
    x = x[:n_samples]
    x_min, _ = torch.min(torch.flatten(x, start_dim=2), dim=-1)
    x_max, _ = torch.max(torch.flatten(x, start_dim=2), dim=-1)
    x_min = x_min[..., None, None]
    x_max = x_max[..., None, None]
    x = (x - x_min) / (x_max - x_min)
    writer.add_images(tag=f"sample_batch_{tag}", img_tensor=x)


def add_hyperparameters(config: dict):
    """Add hyperparameters to Tensorboard."""


def add_hist_params(
    model: nn.Module,
    writer: SummaryWriter,
    global_step: int,
    add_weights: bool = True,
    add_bias: bool = True,
) -> None:
    """Add histogram for trainable parameters such as weights and biases to Tensorboard.

    Allows to determine whether parameters are sufficiently updated.
    Especially in case of vanishing gradients.

    Args:
        model:
        writer:
        global_step:
        add_weights:
        add_bias:

    """
    for name, module in model.named_modules():

        if add_weights:
            if hasattr(module, "weight"):
                if module.weight is not None:
                    writer.add_histogram(
                        tag=f"{name}_weight",
                        values=module.weight.data,
                        global_step=global_step,
                    )
        if add_bias:
            if hasattr(module, "bias"):
                if module.bias is not None:
                    writer.add_histogram(
                        tag=f"{name}_bias",
                        values=module.bias.data,
                        global_step=global_step,
                    )


def add_hparams(
    writer: SummaryWriter, 
    config: Config, 
    train_loss: float, 
    train_accuracy: float, 
    test_loss: float, 
    test_accuracy: float
) -> None:
    """Adds hyperparameters to tensorboard."""

    hparam_dict = {
        "trainer/learning_rate": config.trainer.learning_rate,
        "trainer/batch_size": config.trainer.batch_size,
        "trainer/n_epochs": config.trainer.n_epochs,
        "trainer/weight_decay": config.trainer.weight_decay,
    }

    metric_dict = {
        "hparam/train_loss": train_loss,
        "hparam/train_accuracy": train_accuracy,
        "hparam/test_loss": test_loss,
        "hparam/test_accuracy": test_accuracy,
    }

    writer.add_hparams(hparam_dict, metric_dict) 