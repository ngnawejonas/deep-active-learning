import gc
import copy
from json import load
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from train_utils import get_attack_fn

def get_optimizer(name):
    if name.lower() == 'rmsprop':
        return torch.optim.RMSprop
    elif name.lower() == 'sgd':
        return torch.optim.SGD
    elif name.lower() == 'adam':
        return torch.optim.Adam
    else:
        raise NotImplementedError

def get_scheduler(name):
    if name.lower() == 'cycliclr':
        opt = torch.optim.lr_scheduler.CyclicLR
    elif name.lower() =='cosineannealinglr':
        opt = torch.optim.lr_scheduler.CosineAnnealingLR
    else:
        raise NotImplementedError
    return opt

class Net:
    def __init__(self, net, params, device):
        self.net = net
        self.clf = None
        self.params = params
        self.device = device

    def train(self, data):
        if self.params['repeat'] > 0:
            self._train_xtimes(data)
        else:
            self._train_once(data)

    def train_step(self, epoch, train_loader, optimizer):
        for x, y, idxs in train_loader:
            x, y = x.to(self.device), y.to(self.device)
            optimizer.zero_grad()
            if self.params['advtrain_mode']:
                attack_name = self.params['train_attack']['name']
                attack_params = self.params['train_attack']['args']
                if attack_params.get('norm'):
                    attack_params['norm'] = np.inf if attack_params['norm']=='np.inf' else 2
                attack_fn = get_attack_fn(attack_name)
                x = attack_fn(self.clf, x, **attack_params)
            out = self.clf(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            optimizer.step()

    def val_step(self,epoch):
        pass

    def _train_once(self, data):
        n_epoch = self.params['epochs']
        if self.params['reset'] or not self.clf:
            self.clf = self.net().to(self.device)
            # if self.device.type=='cuda':
            #     self.clf = nn.DataParallel(self.clf)
        self.clf.train()  # set train mode
        optimizer_ = get_optimizer(self.params['optimizer']['name'])
        optimizer = optimizer_(
            self.clf.parameters(),
            **self.params['optimizer']['params'])
        scheduler_ = get_scheduler(self.params["scheduler"]["name"])
        scheduler = scheduler_(optimizer, **self.params["scheduler"]["params"])
        train_loader = DataLoader(data, shuffle=True, **self.params['train_loader_args'])

        def train_step_(self, epoch):
            for x, y, idxs in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                if self.params['advtrain_mode']:
                    attack_name = self.params['train_attack']['name']
                    attack_params = self.params['train_attack']['args']
                    if attack_params.get('norm'):
                        attack_params['norm'] = np.inf if attack_params['norm']=='np.inf' else 2
                    attack_fn = get_attack_fn(attack_name)
                    x = attack_fn(self.clf, x, **attack_params)
                out = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()

        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            # print('==============epoch: %d, lr: %.3f==============' % (epoch, scheduler.get_lr()[0]))
            train_step_(epoch)
            self.val_step(epoch)
            scheduler.step()
        # Clear GPU memory in preparation for next model training
        gc.collect()
        torch.cuda.empty_cache()

    def noval_train_xtimes(self, data):
        """train x times with full data."""

        n_epoch = self.params['n_epoch']
        n_train = (int)(len(data) * 0.8)

        best_model = None
        best_loss = np.inf
        for i in range(self.params['repeat']):
            print(f'training No {i+1}')

            self.clf = self.net().to(self.device)
            # if self.device.type=='cuda':
            #     self.clf = nn.DataParallel(self.clf)

            self.clf.train()  # set train mode
            optimizer_ = get_optimizer(self.params['optimizer'])
            optimizer = optimizer_(
                self.clf.parameters(),
                **self.params['optimizer_args'])

            train_loader = DataLoader(
                data,
                shuffle=True,
                **self.params['train_args'])
            for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
                self.train_step(epoch, train_loader, optimizer)
                self.val_step(epoch)
                # scheduler.step()

            train_loss = self.predict_loss(data)

            if train_loss < best_loss:
                best_loss = train_loss
                best_model = copy.deepcopy(self.clf)
            # Clear GPU memory in preparation for next model training
            gc.collect()
            torch.cuda.empty_cache()
        self.clf = best_model


    def _train_xtimes(self, data):
        """train x times."""

        n_epoch = self.params['epochs']
        n_train = (int)(len(data) * 0.8)

        best_model = None
        best_loss = np.inf
        for i in range(self.params['repeat']):
            print(f'training No {i+1}')
            # shuffle and split data into train and val
            train_data, val_data = random_split(
                data, [n_train, len(data) - n_train])

            self.clf = self.net().to(self.device)
            # if self.device.type=='cuda':
            #     self.clf = nn.DataParallel(self.clf)

            self.clf.train()  # set train mode
            optimizer_ = get_optimizer(self.params['optimizer'])
            optimizer = optimizer_(
                self.clf.parameters(),
                **self.params['optimizer_args'])

            train_loader = DataLoader(
                train_data,
                shuffle=True,
                **self.params['train_args'])
            for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
                self.train_step(epoch, train_loader, optimizer)
                self.val_step(epoch)
                # scheduler.step()
            validation_loss = self.predict_loss(val_data)

            if validation_loss < best_loss:
                best_loss = validation_loss
                best_model = copy.deepcopy(self.clf)
            # Clear GPU memory in preparation for next model training
            gc.collect()
            torch.cuda.empty_cache()
        self.clf = best_model

    def predict_example(self, x):
        if len(x.shape) < 4:
            x = x.unsqueeze(0)
        self.clf.eval()
        with torch.no_grad():
            x = x.to(self.device)
            out = self.clf(x)
            pred = out.max(1)[1]
        return pred.cpu()

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['train_loader_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
            # for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds

    def predict_adv(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for x, y, idxs in loader:
        # for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            attack_name = self.params['test_attack']['name']
            attack_params = self.params['test_attack']['args']
            if attack_params.get('norm'):
                attack_params['norm'] = np.inf if attack_params['norm']=='np.inf' else 2
            attack_fn = get_attack_fn(attack_name)
            x = attack_fn(self.clf, x, **attack_params)
            out = self.clf(x)
            pred = out.max(1)[1]
            preds[idxs] = pred.cpu()
        return preds

    def predict_prob(self, data):
        self.clf.eval()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu()
        probs /= n_drop
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        self.clf.train()
        probs = torch.zeros([n_drop, len(data), len(np.unique(data.Y))])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():
                for x, y, idxs in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    out = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu()
        return probs

    def get_embeddings(self, data):
        self.clf.eval()
        embeddings = torch.zeros([len(data), self.clf.get_embedding_dim()])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                embeddings[idxs] = self.clf.get_embedding().cpu()
        return embeddings

    def predict_loss(self, data):
        self.clf.eval()
        loss = 0.
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        num_batches = len(loader)
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                loss += F.cross_entropy(out, y)

        return loss / num_batches


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

    # def reset_model_weights(self):
    #     for m in (self.embedding, self.fc_head):
    #         for layer in m.children():
    #            if hasattr(layer, 'reset_parameters'):
    #                layer.reset_parameters()

#https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/
#Defining the convolutional neural network 
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 10
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc_head = nn.Linear(84, num_classes)

    def embedding(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        return out

    def forward(self, x):
        self.e1 = self.embedding(x)
        out = self.fc_head(self.e1)
        return out

# https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
# class LeNet5(nn.Module):

#     def __init__(self):
#         super().__init__()
        
#         n_classes = 10

#         self.embedding = nn.Sequential(            
#             nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
#             nn.Tanh(),
#             nn.AvgPool2d(kernel_size=2),
#             nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
#             nn.Tanh(),
#             nn.Linear(in_features=120, out_features=84),
#             nn.Tanh()
#         )
#         self.fc_head = nn.Linear(in_features=84, out_features=n_classes)

#     def forward(self, x):
#         self.e1 = self.embedding(x) 
#         x = torch.flatten(self.e1, 1)
#         x = self.fc_head(x)
#         return x

#     def get_embedding_dim(self):
#         return self.fc_head[0].in_features


class MNIST_Net(TORCHVISION_Net):
    def __init__(self):
        n_classes = 10
        # img_size = (1, 28, 28)  # CHW
        model = models.resnet18(num_classes=n_classes)
        model.conv1 = nn.Conv2d(
            1, 64, kernel_size=(
                7, 7), stride=(
                2, 2), padding=(
                3, 3), bias=False)
        super().__init__(model)

class SVHN_Net(TORCHVISION_Net):
    def __init__(self):
        n_classes = 10
        model = models.resnet18(num_classes=n_classes)
        super().__init__(model)


class CIFAR10_Net(TORCHVISION_Net):
    def __init__(self):
        n_classes = 10
        model = models.resnet18(num_classes=n_classes)
        # model = ResNet18()
        super().__init__(model)

class CIFAR10_Net2(TORCHVISION_Net):
    def __init__(self):
        n_classes = 10
        model = models.vgg16(num_classes=n_classes)
        super().__init__(model)


class oMNIST_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        self.e1 = F.relu(self.fc1(x))
        x = F.dropout(self.e1, training=self.training)
        x = self.fc2(x)
        return x

    def get_embedding_dim(self):
        return 50