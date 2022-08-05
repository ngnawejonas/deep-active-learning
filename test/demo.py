import time
import os
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

from demo_models import CIFAR10_Net, MNIST_Net
from demo_data import get_CIFAR10, get_MNIST
from tqdm import tqdm

import wandb

# import tensorflow as tf

PARAMS = {'n_epoch': 200,
          'train_args': {'batch_size': 64, 'num_workers': 0},
          'test_args': {'batch_size': 1000, 'num_workers': 0},
          'optimizer': 'SGD',
          'optimizer_args': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.0005}
          }

PATH = "checkpoints/model_epoch_{}.pt"
SAVE_EVERY = 20


def get_optimizer(name):
    if name.lower() == 'rmsprop':
        opt = optim.RMSprop
    elif name.lower() == 'sgd':
        opt = optim.SGD
    elif name.lower() == 'adam':
        opt = optim.Adam
    else:
        raise NotImplementedError
    return opt





def load_checkpoint(epoch):
    checkpoint = torch.load(PATH.format(epoch))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def train(clf, data, device):
    # tf_summary_writer = tf.summary.create_file_writer('tfboard')
    n_epoch = PARAMS['n_epoch']
    clf = clf.to(device)
    clf.train()  # set train mode
    optimizer_ = get_optimizer(PARAMS['optimizer'])
    optimizer = optimizer_(
        clf.parameters(),
        **PARAMS['optimizer_args'])
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=0.001,
        step_size_up=20,
        max_lr=0.1,
        mode='triangular2')

    train_accuracy = torchmetrics.Accuracy()
    val_accuracy = torchmetrics.Accuracy()

    loader = DataLoader(data, shuffle=True, **PARAMS['train_args'])
    for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
        # print('==============epoch: %d, lr: %.3f==============' % (epoch, scheduler.get_lr()[0]))
        for x, y in loader:
            if len(x.shape)> 4:
                x, y = x.squeeze(1).to(device), y.squeeze(1).to(device)
            else:
                x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = clf(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()
            #with tf_summary_writer.as_default():
            #    tf.summary.scalar('loss', loss.detach().numpy(), step=step)
            #    step = step + 1
        scheduler.step()
        if (epoch+1)%SAVE_EVERY == 0:
            EPOCH = 5
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': clf.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss.detach().cpu().numpy(),
                        }, PATH.format(epoch))
            val_acc = test(clf, val_data, val_accuracy, device)
            wandb.log({'val acc': val_acc})
        wandb.log({'train loss': loss.detach().cpu().numpy()})


def test(clf, data, metric, device):
    clf = clf.to(device)
    clf.eval()
    loader = DataLoader(data, shuffle=False, **PARAMS['test_args'])
    with torch.no_grad():
        for x, y in loader:
            if len(x.shape)> 4:
                x, y = x.squeeze(1).to(device), y.squeeze(1).to(device)
            else:
                x, y = x.to(device), y.to(device)
            out = clf(x)
            # pred = out.max(1)[1]
            metric.update(out, data.targets[idx])
    acc = metric.compute()
    metric.reset()
    return acc




if __name__ == "__main__":
    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=4000, help='data size')
    args = parser.parse_args()

    #
    wandb.init(project="demo_debug")
    #
    ckpath = 'checkpoints'
    if not os.path.exists(ckpath):
        os.makedirs(ckpath)
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
    print(f'getting dataset...: size={args.n}')
    train_data, test_data = get_CIFAR10(args.n)        # load dataset
    # print('dataset loaded')
    net = CIFAR10_Net()           # load network models.resnet18(num_classes=n_classes)

    # start experiment
    print()
    start = time.time()
    train(net, train_data, device)
    print("train time: {:.2f} s".format(time.time() - start))
    print('testing...')
    acc = test(net, test_data, device)

    print(f"Test accuracy: {acc}")

    T = time.time() - start
    print(f'Total time: {T/60:.2f} mins.')
