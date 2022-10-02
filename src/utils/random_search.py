"""Random search module.

Creates random sets of hyperparameters based
on hparams.yml configuration file.

"""
import random
import time

from ..config.config import Config


def create_random_config_(config: Config) -> None:
    """Creates random confuguration."""

    random.seed(time.time())

    # Learning rate:
    learning_rate = 10 ** random.uniform(-5, -3)
    config.trainer.learning_rate = learning_rate

    # Batch size
    batch_size = random.choice([16, 32, 64, 128])
    config.trainer.batch_size = batch_size

    # Number of transformer blocks
    n_blocks = random.choice([1, 2, 3, 4])
    config.densenet.n_blocks = n_blocks

    n_dims_hidden = random.choice([32, 64, 128, 256, 512])
    config.densenet.n_dims_hidden = n_dims_hidden
