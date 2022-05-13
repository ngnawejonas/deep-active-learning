import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.handler = handler

        self.n_pool = len(X_train)
        self.n_test = len(X_test)

        self.labeled_idxs = np.zeros(self.n_pool, dtype=bool)
        # To handle addition of adversarial dataset to labelled pool
        self.X_train_extra = torch.Tensor([])
        self.Y_train_extra = torch.Tensor([])

    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        if len(self.X_train_extra) > 0:
            X = torch.vstack([self.X_train[labeled_idxs], self.X_train_extra])
            Y = torch.hstack([self.Y_train[labeled_idxs], self.Y_train_extra])
        else:
            X = self.X_train[labeled_idxs]
            Y = self.Y_train[labeled_idxs]
        return labeled_idxs, self.handler(X, Y)

    def n_labeled(self):
        return sum(self.labeled_idxs) + len(self.X_train_extra)

    def get_unlabeled_data(self, n_subset=None):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        if n_subset:
            unlabeled_idxs = unlabeled_idxs[:n_subset]
        return unlabeled_idxs, self.handler(
            self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)

    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test == preds).sum().item() / self.n_test


def get_xMNIST(x_fn, handler, pool_size):
    raw_train = x_fn(root='data', train=True, download=True)
    raw_test = x_fn(root='data', train=False, download=True)
    X_train = raw_train.data[:pool_size]
    Y_train = raw_train.targets[:pool_size]
    X_test =  raw_test.data[:pool_size]
    Y_test = raw_test.targets[:pool_size]
    return Data(X_train, Y_train, X_test, Y_test, handler)


def get_MNIST(handler, pool_size):
    return get_xMNIST(datasets.MNIST, handler, pool_size)


def get_FashionMNIST(handler, pool_size):
    return get_xMNIST(datasets.FashionMNIST, handler, pool_size)


def get_SVHN(handler, pool_size):
    data_train = datasets.SVHN('data', split='train', download=True)
    data_test = datasets.SVHN('data', split='test', download=True)
    return Data(data_train.data[:pool_size],
                torch.from_numpy(data_train.labels)[:pool_size],
                data_test.data[:pool_size],
                torch.from_numpy(data_test.labels)[:pool_size],
                handler)


def get_CIFAR10(handler, pool_size):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data[:pool_size],
                torch.LongTensor(data_train.targets)[:pool_size],
                data_test.data[:pool_size],
                torch.LongTensor(data_test.targets)[:pool_size],
                handler)
