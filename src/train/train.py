import datetime
import os
import time

import torch
from torch import nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

from ..utils.stats import comp_stats_classification
from ..summary.summary import add_graph, add_input_samples, add_hist_params, add_hparams
from ..config.config import Config


def train(model: torch.nn.Module, dataloader: tuple, config: Config) -> None:
    """Train method.

    Args:
        model: PyTorch model.
        dataloader: Tuple holding training and test dataloader.
        config: Class holding configuration.
    """
    runs_dir = config.dirs.runs
    dataset = config.dataloader.dataset

    uid = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.%f")
    tag = config.tag

    log_dir = os.path.join(runs_dir, f"{uid}_{dataset}{f'_{tag}' if tag else ''}")

    writer = SummaryWriter(log_dir=log_dir)
    run_training(model=model, dataloader=dataloader, writer=writer, config=config)
    writer.close()


def run_training(model, dataloader, writer, config: Config) -> None:
    """Main training logic.

    Trains passed model with data coming from dataloader.

    Args:
        model: PyTorch model.
        dataloader: Training and test data loader.
        writer: Tensorboard writer instance.
        config: Class holding configuration.
    """
    tag = config.tag
    device = config.trainer.device
    dataset = config.dataloader.dataset
    n_epochs = config.trainer.n_epochs
    save_train_stats_every_n_epochs = config.summary.save_train_stats_every_n_epochs
    save_test_stats_every_n_epochs = config.summary.save_test_stats_every_n_epochs
    step_size = config.trainer.lr_step_size
    gamma = config.trainer.lr_gamma

    trainloader, testloader = dataloader

    # Add graph of model to Tensorboard.
    if config.summary.add_graph:
        add_graph(model=model, dataloader=trainloader, writer=writer, config=config)

    # Add sample batch to Tensorboard.
    if config.summary.add_sample_batch:
        add_input_samples(
            dataloader=trainloader, writer=writer, tag="train", global_step=0
        )
        add_input_samples(
            dataloader=testloader, writer=writer, tag="test", global_step=0
        )

    learning_rate = config.trainer.learning_rate
    weight_decay = config.trainer.weight_decay

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

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
            running_accuracy += (torch.argmax(outputs, dim=1) == labels).float().sum()
            running_counter += labels.size(0)

        writer.add_scalar("time_per_epoch", time.time() - t0, global_step=epoch)
        writer.add_scalar(
            "learning_rate", scheduler.get_last_lr()[0], global_step=epoch
        )

        scheduler.step()

        train_loss = running_loss / running_counter
        train_accuracy = running_accuracy / running_counter

        if (epoch % save_train_stats_every_n_epochs == 0) or (epoch + 1 == n_epochs):
            writer.add_scalar("train_loss", train_loss, global_step=n_update_steps)
            writer.add_scalar(
                "train_accuracy", train_accuracy, global_step=n_update_steps
            )

        if (epoch % save_test_stats_every_n_epochs == 0) or (epoch + 1 == n_epochs):
            test_loss, test_accuracy = comp_stats_classification(
                model=model, criterion=criterion, data_loader=testloader, device=device
            )
            writer.add_scalar("test_loss", test_loss, global_step=n_update_steps)
            writer.add_scalar(
                "test_accuracy", test_accuracy, global_step=n_update_steps
            )

            if config.summary.add_hparams:
                add_hparams(
                    writer, config, train_loss, train_accuracy, test_loss, test_accuracy
                )

        if config.summary.add_params_hist_every_n_epochs > 0:
            if (epoch % config.summary.add_params_hist_every_n_epochs == 0) or (
                epoch + 1 == n_epochs
            ):
                add_hist_params(model=model, writer=writer, global_step=epoch)

        if config.summary.save_model_every_n_epochs > 0:
            if (epoch % config.summary.save_model_every_n_epochs == 0) or (
                epoch + 1 == n_epochs
            ):
                model_name = (
                    f"{dataset}_epoch_{epoch:04d}{f'_{tag}' if tag else ''}.pth"
                )
                model_path = os.path.join(config.dirs.weights, model_name)
                torch.save(model.state_dict(), model_path)

        print(f"{epoch:04d} {train_loss:.5f} {train_accuracy:.4f}")
