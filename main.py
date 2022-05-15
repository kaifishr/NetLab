"""Neural network testbed for rapid testing of ideas.
"""
import json

from src.modules.model import ConvNet, DenseNet
from src.data.dataloader import get_dataloader
from src.config.config import init_config
from src.train.train import train
from src.utils.tools import set_random_seed


def experiment_imagewoof():

    # Seed random number generator
    set_random_seed(seed=69)

    # Get configuration file
    config = init_config(file_path="config.yml")
    config["data"]["dataset"] = "imagewoof"

    # Get dataloader
    dataloader = get_dataloader(config=config)
    print(json.dumps(config, indent=4))

    # Get the model
    model = ConvNet(config=config)
    model.to(config["device"])

    train(model=model, dataloader=dataloader, config=config)

    print("Experiment finished.")


def experiment_cifar10():

    # Seed random number generator
    set_random_seed(seed=69)

    # Get configuration file
    config = init_config(file_path="config.yml")
    config["data"]["dataset"] = "cifar10"

    # Get dataloader
    dataloader = get_dataloader(config=config)
    print(json.dumps(config, indent=4))

    # Get the model
    model = DenseNet(config=config)
    model.to(config["device"])

    train(model=model, dataloader=dataloader, config=config)

    print("Experiment finished.")


def main():
    # experiment_imagewoof()
    experiment_cifar10()


if __name__ == "__main__":
    main()
