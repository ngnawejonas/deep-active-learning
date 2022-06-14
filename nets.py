import gc
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from train_utils import get_optimizer , EarlyStopping, get_attack_fn, adv_params



class Net:
    def __init__(self, net, params, device, repeat=0, reset=True):
        self.net = net
        self.clf = None
        self.params = params
        self.device = device
        self.reset = reset
        self.repeat = repeat
        self.adv_train_mode = False


    def train(self, data):
        if self.repeat > 0:
            self._train_xtimes(data)
        else:
            self._train_once(data)

    def _train_once(self, data):
        n_epoch = self.params['n_epoch']
        if self.reset or not self.clf:
            self.clf = self.net().to(self.device)
            # if self.device.type=='cuda':
            #     self.clf = nn.DataParallel(self.clf)
        self.clf.train()  # set train mode
        optimizer_ = get_optimizer(self.params['optimizer'])
        optimizer = optimizer_(
            self.clf.parameters(),
            **self.params['optimizer_args'])

        # Early Stopping
        patience = n_epoch//5 if n_epoch//5 > 20 else n_epoch
        early_topping = EarlyStopping(patience=patience)
        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                if self.adv_train_mode:
                    attack_name = adv_params['train_attack']['name']
                    attack_params = adv_params['train_attack']['args']
                    attack_fn = get_attack_fn(attack_name)
                    x = attack_fn(self.clf, x, **attack_params)
                out = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
                early_topping(validation_loss)
                if early_topping.early_stop:
                    break
                # print(f"epoch {epoch}, batch_idx {batch_idx}")
        # Clear GPU memory in preparation for next model training
        gc.collect()
        torch.cuda.empty_cache()

    def _train_xtimes(self, data):
        """train x times."""

        n_epoch = self.params['n_epoch']
        n_train = (int)(len(data) * 0.8)

        best_model = None
        best_loss = np.inf
        for i in range(self.repeat):
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

            # Early Stopping
            # patience = n_epoch//5 if n_epoch//5 > 20 else n_epoch
            # early_topping = EarlyStopping(patience=patience)

            loader = DataLoader(
                train_data,
                shuffle=True,
                **self.params['train_args'])
            for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
                for batch_idx, (x, y, idxs) in enumerate(loader):
                    x, y = x.to(self.device), y.to(self.device)
                    optimizer.zero_grad()
                    out = self.clf(x)
                    loss = F.cross_entropy(out, y)
                    loss.backward()
                    optimizer.step()

            validation_loss = self.predict_loss(val_data)

            if validation_loss < best_loss:
                best_loss = loss
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
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
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
            attack_name = adv_params['test_attack']['name']
            attack_params = adv_params['test_attack']['args']
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
        e1 = self.embedding(x)
        self.e1 = e1
        x = torch.flatten(e1, 1)
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
        super().__init__(model)