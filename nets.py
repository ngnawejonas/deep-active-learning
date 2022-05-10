import gc
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm

from train_utils import get_optimizer #, EarlyStopping



class Net:
    def __init__(self, net, params, device, reset=False):
        self.net = net
        self.clf = None
        self.params = params
        self.device = device
        self.reset = reset

    def train(self, data):
        n_epoch = self.params['n_epoch']
        if self.reset or not self.clf:
            self.clf = self.net().to(self.device)
        self.clf.train()  # set train mode
        optimizer_ = get_optimizer(self.params['optimizer'])
        optimizer = optimizer_(
            self.clf.parameters(),
            **self.params['optimizer_args'])

        # Early Stopping
        # patience = n_epoch//5 if n_epoch//5 > 20 else n_epoch
        # early_topping = EarlyStopping(patience=patience)

        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        for epoch in tqdm(range(1, n_epoch + 1), ncols=100):
            # for batch_idx, (x, y, idxs) in enumerate(loader):
            for x, y, idxs in loader:
            # a = 2
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out, e1 = self.clf(x)
                loss = F.cross_entropy(out, y)
                loss.backward()
                optimizer.step()
                # break
                # early_topping(validation_loss)
                # if early_topping.early_stop:
                #     break
                # print(f"epoch {epoch}, batch_idx {batch_idx}")
        # Clear GPU memory in preparation for next model training
        gc.collect()
        torch.cuda.empty_cache()

    def train_xtimes(self, data, repeat=5):
        """train x times."""

        n_epoch = self.params['n_epoch']
        n_train = (int)(len(data) * 0.8)

        best_model = None
        best_loss = np.inf
        for _ in range(REPEAT):
            # log(f'training No {i+1}')
            # shuffle and split data into train and val
            train_data, val_data = random_split(
                data, [n_train, len(data) - n_train])

            self.clf = self.net().to(self.device)
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
                    out, e1 = self.clf(x)
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

    def predict_example(self, data):
        self.clf.eval()
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                pred = out.max(1)[1]
                preds[idxs] = pred.cpu()
        return preds

    def predict(self, data):
        self.clf.eval()
        preds = torch.zeros(len(data), dtype=data.Y.dtype)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
            # for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
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
                out, e1 = self.clf(x)
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
                    out, e1 = self.clf(x)
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
                    out, e1 = self.clf(x)
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
                out, e1 = self.clf(x)
                embeddings[idxs] = e1.cpu()
        return embeddings

    def predict_loss(self, data):
        self.clf.eval()
        loss = 0.
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        num_batches = len(loader)
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                loss += F.cross_entropy(out, y)

        return loss / num_batches


class TORCHVISION_Net(nn.Module):
    def __init__(self, torchv_model):
        super().__init__()
        layers = list(torchv_model.children())
        self.embedding = torch.nn.Sequential(*(layers[:-1]))
        self.fc_head = torch.nn.Sequential(*(layers[-1:]))

    def forward(self, x):
        e1 = self.embedding(x)
        x = torch.flatten(e1, 1)
        x = self.fc_head(x)
        return x, e1

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


class SVHN_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        e1 = F.relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim():
        return 50


class CIFAR10_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim():
        return 50
