import numpy as np
import torch
from torchvision import datasets
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
        self.X_train_extra = np.array([])
        self.Y_train_extra = np.array([])

    def initialize_labels(self, num):
        # generate initial labeled pool
        tmp_idxs = np.arange(self.n_pool)
        np.random.shuffle(tmp_idxs)
        self.labeled_idxs[tmp_idxs[:num]] = True
    
    def get_labeled_data(self):
        labeled_idxs = np.arange(self.n_pool)[self.labeled_idxs]
        if self.X_train_extra.size:
            X = np.vstack([self.X_train[labeled_idxs], self.X_train_extra])
            Y = np.vstack([self.Y_train[labeled_idxs], self.Y_train_extra])
        else:
            X = self.X_train[labeled_idxs]
            Y = self.Y_train[labeled_idxs]
        return labeled_idxs, self.handler(X, Y)
    
    def get_unlabeled_data(self, n_subset=None):
        unlabeled_idxs = np.arange(self.n_pool)[~self.labeled_idxs]
        if n_subset:
            unlabeled_idxs = unlabeled_idxs[:n_subset]
        return unlabeled_idxs, self.handler(self.X_train[unlabeled_idxs], self.Y_train[unlabeled_idxs])
    
    def get_train_data(self):
        return self.labeled_idxs.copy(), self.handler(self.X_train, self.Y_train)
        
    def get_test_data(self):
        return self.handler(self.X_test, self.Y_test)
    
    def cal_test_acc(self, preds):
        return 1.0 * (self.Y_test==preds).sum().item() / self.n_test

    
def get_xMNIST(x_fn, handler):
    raw_train = x_fn(root='data', train=True, download=True, transform=ToTensor())
    raw_test = x_fn(root='data', train=False, download=True, transform=ToTensor())
    dtl = DataLoader(raw_train, batch_size=len(raw_train))
    for X, Y in dtl:
        X_train = X
        Y_train = Y
    dtl = DataLoader(raw_test, batch_size=len(raw_test))
    for X, Y in dtl:
        X_test = X
        Y_test = Y
    return Data(X_train[:50000], Y_train[:50000], X_test[:50000], Y_test[:50000], handler)

def get_MNIST(handler):
    return get_xMNIST(datasets.MNIST, handler)

def get_FashionMNIST(handler):
    return get_xMNIST(datasets.FashionMNIST, handler)

def get_SVHN(handler):
    data_train = datasets.SVHN('data', split='train', download=True)
    data_test = datasets.SVHN('data', split='test', download=True)
    return Data(data_train.data[:50000], torch.from_numpy(data_train.labels)[:50000], data_test.data[:50000], torch.from_numpy(data_test.labels)[:50000], handler)

def get_CIFAR10(handler):
    data_train = datasets.CIFAR10('./data/CIFAR10', train=True, download=True)
    data_test = datasets.CIFAR10('./data/CIFAR10', train=False, download=True)
    return Data(data_train.data[:50000], torch.LongTensor(data_train.targets)[:50000], data_test.data[:50000], torch.LongTensor(data_test.targets)[:50000], handler)
