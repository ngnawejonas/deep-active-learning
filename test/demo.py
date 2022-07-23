from pprint import pprint
import os
import argparse
import time
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

from tqdm import tqdm

import tensorflow as tf

 
PARAMS = {'n_epoch': 200,
           'train_args': {'batch_size': 64, 'num_workers': 0},
           'test_args': {'batch_size': 1000, 'num_workers': 0},
           'optimizer': 'SGD',
           'optimizer_args': {'lr': 0.1, 'momentum': 0.9, 'weight_decay':0.0005}
        }


def get_optimizer(name):
    if name.lower() == 'rmsprop':
        return optim.RMSprop
    elif name.lower() == 'sgd':
        return optim.SGD
    elif name.lower() == 'adam':
        return optim.Adam
    else:
        raise NotImplementedError

def get_CIFAR10():
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

    return data_train, data_test


def train(clf, data, device):
    tf_summary_writer = tf.summary.create_file_writer('tfboard')
    n_epoch = PARAMS['n_epoch']
    clf = clf.to(device)
    clf.train()  # set train mode
    optimizer_ = get_optimizer(PARAMS['optimizer'])
    optimizer = optimizer_(
        clf.parameters(),
        **PARAMS['optimizer_args'])
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001,
                                          step_size_up=20, max_lr=0.1, mode='triangular2')
    loader = DataLoader(data, shuffle=True, **PARAMS['train_args'])
    step = 0
    for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
        # print('==============epoch: %d, lr: %.3f==============' % (epoch, scheduler.get_lr()[0]))
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = clf(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            with tf_summary_writer.as_default():
                tf.summary.scalar('loss', loss.detach().numpy(), step=step)
                step = step + 1
        scheduler.step()


def test(clf, data, device):
    clf.eval()
    preds = torch.zeros(len(data))
    loader = DataLoader(data, shuffle=False, **PARAMS['test_args'])
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = clf(x)
            pred = out.max(1)[1]
            preds[idxs] = pred.cpu()
        acc = 100.0 * (self.Y_test == preds).sum().item() / self.n_test
    return acc

class TORCHVISION_Net(nn.Module):
    def __init__(self, torchv_model):
        super().__init__()
        layers = list(torchv_model.children())
        self.embedding = torch.nn.Sequential(*(layers[:-1]))
        self.fc_head = torch.nn.Sequential(*(layers[-1:]))  
        self.e1 = None

    def forward(self, x):
        self.e1 = self.embedding(x)
        x = torch.flatten(self.e1, 1)
        x = self.fc_head(x)
        return x

    def get_embedding(self):
        if self.e1 is not None:
            return self.e1.squeeze()
        else:
            raise ValueError('Forward should be executed first')

    def get_embedding_dim(self):
        return self.fc_head[0].in_features

class CIFAR10_Net(TORCHVISION_Net):
    def __init__(self):
        n_classes = 10
        model = models.resnet18(num_classes=n_classes)
        super().__init__(model)


if __name__ == "__main__":
    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    
    # fix random seed
    seed = 16301
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using GPU: {use_cuda}')
    # print('getting dataset...')
    train_data, test_data = get_CIFAR10()        # load dataset
    # print('dataset loaded')
    clf = CIFAR10_Net()           # load network models.resnet18(num_classes=n_classes)

    # start experiment
    print()
    start = time.time()
    train(clf, train_data, device)
    print("train time: {:.2f} s".format(time.time() - start))
    print('testing...')
    acc = test()
    print(f"Test accuracy: {acc}")

    T = time.time() - start
    print(f'Total time: {T/60:.2f} mins.')
