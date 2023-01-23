import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
from PIL import Image

from typing import Tuple, Any

# https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/
class MNIST_Handler(Dataset):
    def __init__(self, X, Y, X_extra=None, Y_extra=None, train=True):
        self.X = X
        self.Y = Y
        self.X_extra = X_extra
        self.Y_extra = Y_extra
        if train:
          self.transform=transforms.Compose([ transforms.Resize((32,32)),
                                                  transforms.ToTensor(),
                                                   transforms.Normalize(mean = (0.1307,), std = (0.3081,))])
        else:
          self.transform=transforms.Compose([transforms.Resize((32,32)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean = (0.1325,), std = (0.3105,))])

    def __getitem__(self, index):
      if index < len(self.X):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(x.numpy())
        x = self.transform(x)
      else:
        idx = index - len(self.X)
        x , y = self.Xe[idx], self.Ye[idx]
      return x, y, index

    def __len__(self):
        xlen = len(self.X_extra) if self.X_extra is not None else 0
        return len(self.X) + xlen



class SVHN_Handler(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        x = Image.fromarray(np.transpose(x, (1, 2, 0)))
        x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


# https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
class CIFAR10_Handler(Dataset):
    def __init__(self, X, Y, X_extra=None, Y_extra=None, train=True):
        self.X = X
        self.Y = Y
        self.X_extra = X_extra
        self.Y_extra = Y_extra
        if train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                    ),
                ]
            )

    def __getitem__(self, index):
        if index < len(self.X):
            x, y = self.X[index], self.Y[index]
            x = Image.fromarray(x)
            x = self.transform(x)
        else:
            idx = index - len(self.X)
            x, y = self.X_extra[idx], self.Y_extra[idx]
        x = torch.reshape(x, (3,32,32))
        return x, y, index

    def __len__(self):
        xlen = len(self.X_extra) if self.X_extra is not None else 0
        return len(self.X) + xlen
