# NetLab 

A minimal development framework for rapid prototyping in PyTorch.

NetLab provides boiler plate code for training neural networks 
and lets you focus on developing neural network architectures. 

NetLab comes with a series of useful utilities for rapid prototyping and 
explorative tools.


## Usage

```python
from src.modules.model import ConvNet, DenseNet
from src.data.dataloader import get_dataloader
from src.config.config import init_config
from src.train.train import train
from src.utils.tools import set_random_seed
from src.utils.random_search import create_random_config_


def experiment_imagewoof():

    config = init_config(file_path="config.yml")
    config.dataloader.dataset = "imagewoof"

    set_random_seed(seed=config.random_seed)

    dataloader = get_dataloader(config=config)

    model = ConvNet(config=config)
    model.to(config.trainer.device)

    print(config)
    train(model=model, dataloader=dataloader, config=config)

    print("Experiment finished.")


def main():
    experiment_imagewoof()


if __name__ == "__main__":
    main()
```

## License

MIT
