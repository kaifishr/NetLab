# NetLab 

A minimal development framework for rapid prototyping in PyTorch.

NetLab provides boiler plate code for training neural networks 
and lets you focus on developing neural network architectures. 

NetLab comes with a series of useful utilities for rapid prototyping.


## Usage

```python
import json

from src.modules.model import ConvNet
from src.data.dataloader import get_dataloader
from src.config.config import init_config
from src.train.train import train
from src.utils.tools import set_random_seed


def experiment_imagewoof():

    # Get configuration file
    config = init_config(file_path="config.yml")
    config["data"]["dataset"] = "imagewoof"

    # Seed random number generator
    set_random_seed(seed=config["experiment"]["random_seed"])

    # Get dataloader
    dataloader = get_dataloader(config=config)
    print(json.dumps(config, indent=4))

    # Get the model
    model = ConvNet(config=config)
    model.to(config["device"])

    train(model=model, dataloader=dataloader, config=config)

    print("Experiment finished.")


def main():
    experiment_imagewoof()


if __name__ == "__main__":
    main()
```

## TODOs

- Find better method for configuration (custom config class, argparse, ...)

## License

MIT
