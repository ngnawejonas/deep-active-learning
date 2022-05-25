import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

class Strategy:
    def __init__(self, dataset, net, pseudo_labeling=False):
        self.dataset = dataset
        self.net = net
        self.pseudo_labeling = pseudo_labeling

    def query(self, n):
        pass

    def update(self, pos_idxs, extra_data=None):
        print('query/pos idxs', pos_idxs)
        self.dataset.labeled_idxs[pos_idxs] = True
        if extra_data and self.pseudo_labeling:
            self.add_extra_data(pos_idxs, extra_data)

    def add_extra_data(self, pos_idxs, extra_data):
        if len(extra_data) == 0:
            return
        print('Y_train_extra', self.dataset.Y_train[pos_idxs])
        if len(self.dataset.X_train_extra) > 0:
            self.dataset.X_train_extra = torch.vstack([self.dataset.X_train_extra, extra_data]) 
            self.dataset.Y_train_extra = torch.hstack([self.dataset.Y_train_extra, self.dataset.Y_train[pos_idxs]])
        else:
            self.dataset.X_train_extra = extra_data 
            self.dataset.Y_train_extra = self.dataset.Y_train[pos_idxs]

        print('New Y_train_extra', self.dataset.Y_train_extra)


    def train(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        self.net.train(labeled_data)

    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings

