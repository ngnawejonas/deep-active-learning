import numpy as np
import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


class Data:
    def __init__(self, X_train, Y_train, X_test, Y_test, handler, n_adv_test):
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
        # adv test data
        self.n_adv_test = n_adv_test
        self.adv_test_idxs = np.random.choice(np.arange(self.n_test), self.n_adv_test, replace=False)

    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        # print(tmp_idxs[:5], tmp_idxs[-5:])
        self.labeled_idxs[tmp_idxs[:num]] = True

    def add_extra_data(self, pos_idxs, extra_data):
        # print('Y_train_extra', self.Y_train[pos_idxs])
        if len(self.X_train_extra) > 0:
            self.X_train_extra = torch.vstack([self.X_train_extra, extra_data]) 
            self.Y_train_extra = torch.hstack([self.Y_train_extra, self.Y_train[pos_idxs]])
        else:
            self.X_train_extra = extra_data
            self.Y_train_extra = self.Y_train[pos_idxs]
        # assert len(self.X_train_extra) == len(self.Y_train_extra)
        # print('New Y_train_extra', self.Y_train_extra)

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        if len(self.X_train_extra) > 0:
            # print('data.py:44',self.X_train[labeled_idxs].shape, self.X_train_extra.shape)
            if len(self.X_train_extra.shape) == 3:
                X_train_extra = self.X_train_extra.unsqueeze(1) 
            else:
                X_train_extra = self.X_train_extra
            X = torch.vstack([self.X_train[labeled_idxs], X_train_extra])
            Y = torch.hstack([self.Y_train[labeled_idxs], self.Y_train_extra])
        else:
            X = self.X_train[labeled_idxs]
            Y = self.Y_train[labeled_idxs]
        return labeled_idxs, self.handler(X, Y)

    def n_labeled(self):
        return sum(self.labeled_idxs) + len(self.X_train_extra)

    def get_unlabeled_data(self, n_subset=None):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        np.random.shuffle(unlabeled_idxs)
        if n_subset:
            unlabeled_idxs = unlabeled_idxs[:n_subset]
        X = self.X_train[unlabeled_idxs]
        Y = self.Y_train[unlabeled_idxs]
        return unlabeled_idxs, self.handler(X, Y)

    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)

    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)

    def get_adv_test_data(self):
        return self.handler(self.X_test[self.adv_test_idxs], self.Y_test[self.adv_test_idxs])

    def cal_test_acc(self, preds):
        return 100.0 * (self.Y_test == preds).sum().item() / self.n_test

    def cal_adv_test_acc(self, preds):
        return 100.0 * (self.Y_test[self.adv_test_idxs] == preds).sum().item() / self.n_adv_test


def get_xMNIST(x_fn, handler, pool_size, pref = ''):
    raw_train = x_fn(root='./data/'+pref+'MNIST', train=True, download=True, transform=ToTensor())
    raw_test = x_fn(root='./data/'+pref+'MNIST', train=False, download=True, transform=ToTensor())

    dtl = DataLoader(raw_train, batch_size=len(raw_train))
    for X,y in dtl:
        X_train = X
        Y_train = y

    dtl = DataLoader(raw_test, batch_size=len(raw_test))
    for X,y in dtl:
        X_test = X
        Y_test = y

    # X_train = raw_train.data[:pool_size]
    # Y_train = raw_train.targets[:pool_size]
    # X_test =  raw_test.data[:pool_size]
    # Y_test = raw_test.targets[:pool_size]
    return Data(X_train, Y_train, X_test, Y_test, handler)


def get_MNIST(handler, pool_size):
    return get_xMNIST(datasets.MNIST, handler, pool_size)


def get_FashionMNIST(handler, pool_size):
    return get_xMNIST(datasets.FashionMNIST, handler, pool_size, 'Fashion')


def get_SVHN(handler, pool_size):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
    ])
    data_train = datasets.SVHN('data', split='train', download=True, transform=transform_train)
    data_test = datasets.SVHN('data', split='test', download=True, transform=transform_test)

    dtl = DataLoader(data_train, batch_size=len(data_train))
    for X,y in dtl:
        X_train = X
        Y_train = y

    dtl = DataLoader(data_test, batch_size=len(data_test))
    for X,y in dtl:
        X_test = X
        Y_test = y

    # print('data.py:146 ', X_train.data.shape, X_train.dtype, type(X_train))
    return Data(X_train, Y_train, X_test, Y_test, handler)

# def get_CIFAR10(handler, pool_size):
#     data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
#     data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
#     print('data.py:146 ', data_train.data.shape)
#     return Data(data_train.data[:pool_size],
#                 torch.LongTensor(data_train.targets)[:pool_size],
#                 data_test.data[:pool_size],
#                 torch.LongTensor(data_test.targets)[:pool_size],
#                 handler)

def get_CIFAR10(handler, pool_size, n_adv_test):
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
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True, transform=transform_train)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True, transform=transform_test)

    dtl = DataLoader(data_train, batch_size=len(data_train))
    for X,y in dtl:
        X_train = X
        Y_train = y

    dtl = DataLoader(data_test, batch_size=len(data_test))
    for X,y in dtl:
        X_test = X
        Y_test = y

    # print('data.py:146 ', X_train.data.shape, X_train.dtype, type(X_train))
    return Data(X_train[:pool_size], Y_train[:pool_size], X_test, Y_test, handler, n_adv_test)
