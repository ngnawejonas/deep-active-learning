import numpy as np
import torch
from .strategy import Strategy
from tqdm import tqdm
from torch.utils.data import DataLoader

from train_utils import log_to_file

class AdversarialStrategy(Strategy):
    def __init__(self, dataset, net,
                    repeat = 1,
                    pseudo_labeling=True,
                    n_subset_ul=None,
                    diversity=False, **kwargs):
        super().__init__(dataset, net, repeat, pseudo_labeling)
        self.diversity = diversity
        self.n_subset_ul = n_subset_ul # number of unlabeled data to attack
        self.params = kwargs

    def cal_dis(self, x):
        x_i = x.clone()
        initial_label = self.net.predict_example(x_i)
        # print('dist cal')
        while self.net.predict_example(x_i) == initial_label:
            # print('...attack...')
            x_i = self.attack_fn(x_i.to(self.net.device))
        dis = torch.norm(x_i.cpu() - x)
        return dis.detach(), x_i.detach().squeeze(0)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(self.n_subset_ul)  
        self.net.clf.eval()
        distances = np.zeros(unlabeled_idxs.shape)
        adv_images = []
        iter_loader = iter(DataLoader(unlabeled_data))
        for i in tqdm(range(len(unlabeled_idxs)), ncols=100):
            x, y, idx = iter_loader.next()
            dis, x_adv = self.cal_dis(x)
            log_to_file('distance.txt', f'{i}, {dis.numpy()}')
            distances[i] = dis
            adv_images.append(x_adv.squeeze(0) if x.shape[0]==1 else x_adv)
        selected_idxs = distances.argsort()[:n]
        if self.pseudo_labeling:
            extra_data = torch.stack(adv_images)[selected_idxs]
            return unlabeled_idxs[selected_idxs], extra_data

        return unlabeled_idxs[selected_idxs]

    def add_extra(self, pos_idxs, extra_data):
        if len(self.dataset.X_train_extra) > 0:
            self.dataset.X_train_extra = torch.vstack([self.dataset.X_train_extra, extra_data]) 
            self.dataset.Y_train_extra = torch.hstack([self.dataset.Y_train_extra, self.dataset.Y_train[pos_idxs]])
        else:
            self.dataset.X_train_extra = extra_data 
            self.dataset.Y_train_extra = self.dataset.Y_train[pos_idxs]

    def attack_fn(self, X):
        """attack_fn to be implemented by child classes"""
        pass