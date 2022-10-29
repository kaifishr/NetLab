import os
import tarfile

import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.utils import download_url

from .dataset import TensorDataset


def seed_worker(worker_id):
    """Seed dataloader workers."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader(config: dict) -> tuple[DataLoader, DataLoader]:
    """Creates dataloader for specified dataset."""

    dataset = config.dataloader.dataset 
    num_workers = config.dataloader.num_workers
    batch_size = config.trainer.batch_size

    if dataset == "imagewoof":

        dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof-160.tgz"
        download_url(url=dataset_url, root="./data")

        cwd = os.getcwd()
        with tarfile.open(cwd + "/data/imagewoof-160.tgz", "r:gz") as tar:
            tar.extractall(path=cwd + "/data")

        avg = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)

        stats = (avg, std)

        train_transform = transforms.Compose(
            [
                transforms.Resize(size=160),
                transforms.RandomCrop(size=128),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.5, hue=0.3),
                transforms.RandomRotation(degrees=(0, 45)),
                transforms.RandomErasing(),
                transforms.ToTensor(),
                transforms.Normalize(*stats, inplace=True),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(128),
                transforms.CenterCrop(size=(128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(*stats),
            ]
        )

        data_dir = cwd + "/data/imagewoof-160/"
        train_dataset = torchvision.datasets.ImageFolder(
            root=data_dir + "/train", transform=train_transform
        )

        test_dataset = torchvision.datasets.ImageFolder(
            root=data_dir + "/val", transform=test_transform
        )

        # Add number of classes and input shape to config
        config.data.num_classes = 10
        config.data.input_shape = (3, 128, 128)

    elif dataset == "cifar10":

        cifar10 = torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
        mean = np.mean(np.array(cifar10.data / 255.0), axis=(0, 1, 2))      # [0.49139968 0.48215841 0.44653091]
        std = np.std(np.array(cifar10.data / 255.0), axis=(0, 1, 2))        # [0.24703223 0.24348513 0.26158784]

        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.5, hue=0.3),
                # transforms.RandomRotation(degrees=45),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomErasing(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

        mean, std = None, None

        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )

        # Add number of classes and input shape to config
        config.data.num_classes = 10
        config.data.input_shape = (3, 32, 32)

    elif dataset == "custom_from_numpy":
        """
        Create dataset for classification from Numpy ndarray with
        custom TensorDataset class.

        """

        def get_random_dataset():

            n_samples = 10
            num_classes = 2
            n_dims_in = (3, 32, 32)

            x = np.random.randn(n_dims_in)
            y = np.random.randint(num_classes, size=n_samples)

            x = torch.Tensor(x)
            y = torch.Tensor(y).long()

            return x, y

        train_images, train_labels = get_random_dataset()
        test_images, test_labels = get_random_dataset()

        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(0.5, 0.5),
            ]
        )

        transform_test = transforms.Compose([transforms.Normalize(0.5, 0.5)])

        train_dataset = TensorDataset(
            (train_images, train_labels), transform=transform_train
        )
        test_dataset = TensorDataset(
            (test_images, test_labels), transform=transform_test
        )

        # Add number of classes and input shape to config
        config.data.num_classes = 2
        config.data.input_shape = (3, 32, 32)

    else:
        raise NotImplementedError(f"Dataloader for {dataset} not implemented.")

    generator = torch.Generator()
    generator.manual_seed(config.random_seed)

    trainloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        shuffle=True,
        pin_memory=True,
    )

    testloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=2 * batch_size,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        shuffle=False,
        pin_memory=True,
    )

    return trainloader, testloader
