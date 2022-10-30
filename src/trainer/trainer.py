import datetime
import os
import time

import torch
from torch import nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from ..utils.stats import comp_stats_classification
from ..summary.summary import (
    add_graph,
    add_input_samples,
    add_hist_params,
    add_hparams,
    add_linear_weights,
    add_patch_embedding_weights,
)
from ..config.config import Config
from ..utils.tools import (
    count_model_parameters,
    count_module_parameters,
)


class Trainer:
    """Trainer class.

    Attributes:
        model: PyTorch model.
        dataloader: Tuple holding training and test dataloader.
        config: Class holding configuration.

    Typical usage example:

        model = Modle()
        dataloader = Dataloader()
        config = Config()
        trainer = Trainer(model, dataloader, config):
        trainer.run()
    """

    def __init__(self, model: torch.nn.Module, dataloader: tuple, config: Config):
        """Initializes trainer class."""

        self.model = model
        self.dataloader = dataloader
        self.config = config

        runs_dir = config.dirs.runs
        dataset = config.dataloader.dataset

        uid = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
        tag = config.tag

        log_dir = os.path.join(runs_dir, f"{uid}_{dataset}{f'_{tag}' if tag else ''}")

        count_model_parameters(model=model)
        count_module_parameters(model=model)

        self.writer = SummaryWriter(log_dir=log_dir)

        step_size = config.trainer.lr_step_size
        gamma = config.trainer.lr_gamma
        learning_rate = config.trainer.learning_rate
        weight_decay = config.trainer.weight_decay

        trainloader, testloader = self.dataloader

        if config.summary.add_graph:
            add_graph(
                model=model, dataloader=trainloader, writer=self.writer, config=config
            )

        if config.summary.add_sample_batch:
            add_input_samples(
                dataloader=trainloader, writer=self.writer, tag="train", global_step=0
            )
            add_input_samples(
                dataloader=testloader, writer=self.writer, tag="test", global_step=0
            )

        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )

    def run(self):
        """Main training logic.

        Trains passed model with data coming from dataloader.

        Args:
            model: PyTorch model.
            dataloader: Training and test data loader.
            writer: Tensorboard writer instance.
            config: Class holding configuration.
        """
        device = self.config.trainer.device
        n_epochs = self.config.trainer.n_epochs

        trainloader, testloader = self.dataloader

        config = self.config
        writer = self.writer
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        scheduler = self.scheduler

        n_update_steps = 0

        for epoch in range(n_epochs):

            running_loss = 0.0
            running_accuracy = 0.0
            running_counter = 0

            model.train()
            t0 = time.time()

            for x_data, y_data in trainloader:

                # Get the inputs; data is a list of [inputs, labels]
                inputs, labels = x_data.to(device), y_data.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Feedforward
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Backpropagation
                loss.backward()

                # Gradient descent
                optimizer.step()
                n_update_steps += 1

                # keeping track of statistics
                running_loss += loss.item()
                running_accuracy += (
                    (torch.argmax(outputs, dim=1) == labels).float().sum()
                )
                running_counter += labels.size(0)

            writer.add_scalar("time_per_epoch", time.time() - t0, global_step=epoch)
            writer.add_scalar(
                "learning_rate", scheduler.get_last_lr()[0], global_step=epoch
            )

            scheduler.step()

            train_loss = running_loss / running_counter
            train_accuracy = running_accuracy / running_counter

            summary = config.summary

            if summary.add_train_stats_every_n_epochs:
                if (epoch % summary.add_train_stats_every_n_epochs == 0) or (
                    epoch + 1 == n_epochs
                ):
                    writer.add_scalar(
                        "train_loss", train_loss, global_step=n_update_steps
                    )
                    writer.add_scalar(
                        "train_accuracy", train_accuracy, global_step=n_update_steps
                    )

            if summary.add_test_stats_every_n_epochs:
                if (epoch % summary.add_test_stats_every_n_epochs == 0) or (
                    epoch + 1 == n_epochs
                ):
                    stats = comp_stats_classification(
                        model=model,
                        criterion=criterion,
                        data_loader=testloader,
                        device=device,
                    )
                    test_loss, test_accuracy = stats
                    writer.add_scalar(
                        "test_loss", test_loss, global_step=n_update_steps
                    )
                    writer.add_scalar(
                        "test_accuracy", test_accuracy, global_step=n_update_steps
                    )

                    if summary.add_hparams:
                        add_hparams(
                            writer,
                            config,
                            train_loss,
                            train_accuracy,
                            test_loss,
                            test_accuracy,
                        )

            if summary.add_params_hist_every_n_epochs:
                if (epoch % summary.add_params_hist_every_n_epochs == 0) or (
                    epoch + 1 == n_epochs
                ):
                    add_hist_params(model=model, writer=writer, global_step=epoch)

            if summary.add_model_every_n_epochs:
                if (epoch % summary.add_model_every_n_epochs == 0) or (
                    epoch + 1 == n_epochs
                ):
                    dataset = config.dataloader.dataset
                    tag = f"_{config.tag}" if config.tag else ""
                    model_name = f"{dataset}_epoch_{epoch:04d}{tag}.pth"
                    model_path = os.path.join(config.dirs.weights, model_name)
                    torch.save(model.state_dict(), model_path)

            if summary.add_weights_every_n_epochs:
                if (epoch % summary.add_weights_every_n_epochs == 0) or (
                    epoch + 1 == n_epochs
                ):
                    add_linear_weights(model=model, writer=writer, global_step=epoch)
                    add_patch_embedding_weights(
                        model=model, writer=writer, global_step=epoch
                    )

            print(f"{epoch:04d} {train_loss:.5f} {train_accuracy:.4f}")

        self.writer.close()
