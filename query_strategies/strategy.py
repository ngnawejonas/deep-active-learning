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
        # print('query/pos idxs', pos_idxs)
        nx = len(extra_data) if extra_data is not None else 0
        Nx = nx + len(pos_idxs)
        N1 = Nx + self.dataset.n_labeled()
        
        self.dataset.labeled_idxs[pos_idxs] = True
        if extra_data is not None and self.pseudo_labeling:
            self.dataset.add_extra_data(pos_idxs, extra_data)
        
        N2 = self.dataset.n_labeled()
        print(f'{n}+{len(pos_idxs)}={Nx} added! Now {N2} vs {N1}')

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

