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
from demo_handlers import CIFAR10_Train_Handler, CIFAR10_Test_Handler

from tqdm import tqdm


def get_CIFAR10(n=4000):
    train_data = datasets.CIFAR10(
        './data/CIFAR10',
        train=True,
        download=True,)
    # transform=transform_train)
    test_data = datasets.CIFAR10(
        './data/CIFAR10',
        train=False,
        download=True,)
    # transform=transform_test)

    n_val = int(0.2 * len(train_data))
    train_data, val_data = random_split(
        train_data, [len(train_data) - n_val, n_val])
    trd = CIFAR10_Train_Handler(train_data.dataset.data[train_data.indices], torch.LongTensor(
        train_data.dataset.targets)[train_data.indices])
    vld = CIFAR10_Test_Handler(val_data.dataset.data[train_data.indices], torch.LongTensor(
        val_data.dataset.targets)[train_data.indices])
    tsd = CIFAR10_Test_Handler(
        test_data.data, torch.LongTensor(
            test_data.targets))
    return trd, vld, tsd


def get_MNIST(n=1000):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
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

    n_val = 5000#int(0.2 * len(train_data))
    train_data, val_data = random_split(
        train_data, [len(train_data) - n_val, n_val])

    return train_data, val_data, test_data
