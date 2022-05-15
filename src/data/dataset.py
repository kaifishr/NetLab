import torch
from torch.utils.data import Dataset


class TensorDataset(Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors: tuple[torch.Tensor, torch.Tensor], transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)
