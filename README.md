# NetLab

> *Quantity breeds quality.*

Machine learning engineering is highly experimental and iterative in nature. *NetLab* is a minimal development framework for rapid prototyping in PyTorch to make sure ideas can be validated quickly.

*NetLab* provides boiler plate code for training neural networks 
and lets you focus on the important aspects of machine learning engineering. *NetLab* also comes with explorative methods and useful utilities.

## Usage

### Training a network:

```python
from src.modules.model import ConvNet
from src.data.dataloader import get_dataloader
from src.config.config import init_config
from src.train.train import train
from src.utils.tools import set_random_seed


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

### Random Hyperparameter Search:

```python
from src.modules.model import DenseNet
from src.data.dataloader import get_dataloader
from src.config.config import init_config
from src.train.train import train
from src.utils.tools import set_random_seed
from src.utils.random_search import create_random_config_


def experiment_random_search():

    n_runs = 1000
    n_epochs = 10

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

        train(model=model, dataloader=dataloader, config=config)


def main():
    experiment_random_search()


if __name__ == "__main__":
    main()
```

## Cleaning up

```console
python make_clean.py --folders data/ runs/ weights/
```

## TODO 

- Add callbacks
- Add confusion matrix

## License

MIT
