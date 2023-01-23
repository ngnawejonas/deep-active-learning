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
        self.X_train_extra = None
        self.Y_train_extra = None
        # adv test data
        self.n_adv_test = n_adv_test
        if self.n_adv_test == self.n_test:
            self.adv_test_idxs = np.arange(self.n_test)
        else:
            self.adv_test_idxs = np.random.choice(
                np.arange(self.n_test), self.n_adv_test, replace=False)

    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True

    def add_extra_data(self, pos_idxs, extra_data):
        if self.X_train_extra is not None:
            self.X_train_extra = torch.vstack([self.X_train_extra, extra_data])
            self.Y_train_extra = torch.hstack([self.Y_train_extra, self.Y_train[pos_idxs]])
        else:
            self.X_train_extra = extra_data
            self.Y_train_extra = self.Y_train[pos_idxs]

    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        X = self.X_train[labeled_idxs] 
        Y = self.Y_train[labeled_idxs]
        return labeled_idxs, self.handler(X, Y, self.X_train_extra, self.Y_train_extra)

    def n_labeled(self):
        xlen = len(self.X_train_extra) if self.X_train_extra is not None else 0
        return sum(self.labeled_idxs) + xlen

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
        return self.handler(self.X_test, self.Y_test, train=False)

    def get_adv_test_data(self):
        return self.handler(self.X_test[self.adv_test_idxs], self.Y_test[self.adv_test_idxs], train=False)

    def cal_test_acc(self, preds):
        return 100.0 * (self.Y_test == preds).sum().item() / self.n_test

    def cal_adv_test_acc(self, preds):
        return 100.0 * (self.Y_test[self.adv_test_idxs] == preds).sum().item() / self.n_adv_test

#########################################################################################################
#########################################################################################################
def get_CIFAR10(handler, pool_size, n_adv_test):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data[:pool_size], torch.LongTensor(data_train.targets)[:pool_size], data_test.data[:pool_size], torch.LongTensor(data_test.targets)[:pool_size], handler, n_adv_test)

def get_xMNIST(x_fn, handler, pool_size, n_adv_test, pref=''):
    data_train = x_fn(root='./data/'+pref+'MNIST', train=True,
                      download=True, transform=ToTensor())
    data_test = x_fn(root='./data/'+pref+'MNIST', train=False,
                     download=True, transform=ToTensor())
    return Data(data_train.data[:pool_size], torch.LongTensor(data_train.targets)[:pool_size], data_test.data[:pool_size], torch.LongTensor(data_test.targets)[:pool_size], handler, n_adv_test)


def get_MNIST(handler, pool_size, n_adv_test):
    return get_xMNIST(datasets.MNIST, handler, pool_size, n_adv_test)


def get_FashionMNIST(handler, pool_size):
    return get_xMNIST(datasets.FashionMNIST, handler, pool_size, 'Fashion')


def get_SVHN(handler):
    data_train = datasets.SVHN('./data/SVHN', split='train', download=True)
    data_test = datasets.SVHN('./data/SVHN', split='test', download=True)
    return Data(data_train.data[:40000], torch.from_numpy(data_train.labels)[:40000], data_test.data[:40000], torch.from_numpy(data_test.labels)[:40000], handler)