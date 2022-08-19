import os

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
import torchmetrics
from ray import tune
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split
import wandb
from tqdm import tqdm


PATH = "checkpoints/model_epoch_{}.pt"
SAVE_EVERY = 1

# PARAMS = {'n_epochs': 200,
#           'train_args': {'batch_size': 64, 'num_workers': 4},
#           'test_args': {'batch_size': 1000, 'num_workers': 4},
#           'optimizer': 'SGD',
#           'optimizer_args': {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 0.0005}
#           }

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

def get_scheduler(name):
    if name.lower() == 'cycliclr':
        opt = optim.lr_scheduler.CyclicLR
    elif name.lower() =='cosineannealinglr':
        opt = optim.lr_scheduler.CosineAnnealingLR
    else:
        raise NotImplementedError
    return opt

def load_checkpoint(model, optimizer, epoch):
    checkpoint = torch.load(PATH.format(epoch))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def train(clf, train_data, val_data,config, params, device):
    n_epochs = config['epochs']
    clf = clf.to(device)
    clf.train()  # set train mode
    optimizer_ = get_optimizer(config['optimizer'])
    optimizer = optimizer_(
        clf.parameters(), lr=config["lr_schedule"]["initial_lr"],
        **params['optimizer_args'])
    scheduler_ = get_scheduler(config["lr_schedule"]["scheduler"])
    scheduler = scheduler_(
        optimizer, **config["lr_schedule"]["params"])

    train_accuracy = torchmetrics.Accuracy().to(device)
    val_accuracy = torchmetrics.Accuracy().to(device)

    loader = DataLoader(train_data, shuffle=True, **params['train_loader_args'])
    for epoch in tqdm(range(1, n_epochs + 1), ncols=100):
        # print('==============epoch: %d, lr: %.3f==============' % (epoch, scheduler.get_lr()[0]))
        for x, y in loader:
            if len(x.shape) > 4:
                x, y = x.squeeze(1).to(device), y.squeeze(1).to(device)
            else:
                x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = clf(x)
            loss = torch.nan_to_num(F.cross_entropy(out, y))
            loss.backward()
            optimizer.step()
        scheduler.step()
        if (epoch + 1) % SAVE_EVERY == 0:
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': clf.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': loss.detach().cpu().numpy(),
            # }, PATH.format(epoch))
            val_acc = test(clf, val_data, val_accuracy, params, device)
            tune.report(val_acc=val_acc)
            wandb.log({'val_acc': val_acc})
        wandb.log({'train_loss': loss.detach().cpu().numpy()})


def test(clf, data, metric, params, device):
    clf = clf.to(device)
    clf.eval()
    metric.reset()
    loader = DataLoader(data, shuffle=False, **params['test_loader_args'])
    with torch.no_grad():
        for x, y in loader:
            if len(x.shape) > 4:
                x, y = x.squeeze(1).to(device), y.squeeze(1).to(device)
            else:
                x, y = x.to(device), y.to(device)
            out = clf(x)
            # pred = out.max(1)[1]
            metric.update(out, y)
    acc = metric.compute()
    metric.reset()
    return acc
