import time

import argparse

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torchmetrics
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split

from tqdm import tqdm


def get_CIFAR10(n=1000):
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            ),
        ]
    )
    train_data = datasets.CIFAR10(
        './data/CIFAR10',
        train=True,
        download=True,
        transform=transform_train)
    test_data = datasets.CIFAR10(
        './data/CIFAR10',
        train=False,
        download=True,
        transform=transform_test)

    n_val = int(0.2 * len(train_data))
    train_data, val_data = random_split(train_data, [len(train_data) - n_val, n_val])

    return train_data, test_data


def get_MNIST(n=1000):
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
	train_data = datasets.MNIST(
	    './data/MNIST',
	    train=True,
	    download=True,
	    transform=transform)
	test_data = datasets.MNIST(
	    './data/MNIST',
	    train=False,
	    download=True,
	    transform=transform)

	n_val = int(0.2 * len(train_data))
	train_data, val_data = random_split(train_data, [len(train_data) - n_val, n_val])

	return train_data, test_data