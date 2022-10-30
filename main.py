"""Neural network testbed for rapid testing of ideas.

This file shows a few examples how to use NetLab.
"""
from src.modules.model import ConvNet, DenseNet
from src.data.dataloader import get_dataloader
from src.config.config import init_config
from src.trainer.trainer import Trainer
from src.utils.tools import set_random_seed
from src.utils.random_search import create_random_config_


def experiment_imagewoof():

    config = init_config(file_path="config.yml")

    config.tag = "diffnet"
    config.dataloader.dataset = "imagewoof"

    set_random_seed(seed=config.random_seed)

    dataloader = get_dataloader(config=config)

    model = ConvNet(config=config)
    model.to(config.trainer.device)

    print(config)
    trainer = Trainer(model=model, dataloader=dataloader, config=config)
    trainer.run()

    print("Experiment finished.")


def experiment_cifar10():

    config = init_config(file_path="config.yml")

    config.tag = ""
    config.dataloader.dataset = "cifar10"

    set_random_seed(seed=config.random_seed)

    dataloader = get_dataloader(config=config)

    model = DenseNet(config=config)
    model.to(config.trainer.device)

    print(config)
    trainer = Trainer(model=model, dataloader=dataloader, config=config)
    trainer.run()

    print("Experiment finished.")


def experiment_random_search():

    n_runs = 5
    n_epochs = 1

    config = init_config(file_path="config.yml")

    config.trainer.n_epochs = n_epochs
    config.dataloader.dataset = "cifar10"
    config.tag = "random_search"

    for _ in range(n_runs):

        create_random_config_(config)

        set_random_seed(seed=config.random_seed)

        dataloader = get_dataloader(config=config)

        print(config)
        model = DenseNet(config=config)
        model.to(config.trainer.device)

        trainer = Trainer(model=model, dataloader=dataloader, config=config)
        trainer.run()

    print("Experiment finished.")


def main():
    experiment_imagewoof()
    experiment_cifar10()
    experiment_random_search()


if __name__ == "__main__":
    main()
