import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy
from tqdm import tqdm

class AdversarialStrategy(Strategy):
    def __init__(self, dataset, net,
                    n_subset_ul=None,
                    pseudo_labeling=True,
                    diversity=False, **kwargs):
        super().__init__(dataset, net)
        self.diversity = diversity
        self.n_subset_ul = n_subset_ul # number of unlabeled data to attack
        self.pseudo_labeling = pseudo_labeling
        self.params = kwargs

    def cal_dis(self, x):
        nx = torch.unsqueeze(x, 0)
        
        out, e1 = self.net.clf(nx)
        py = out.max(1)[1]
        ny = out.max(1)[1]
        while py.item() == ny.item():
            nx = self.attack_fn(x=nx, **self.params)

            out, e1 = self.net.clf(nx)
            py = out.max(1)[1]

        dis = torch.norm(nx - torch.unsqueeze(x, 0))
        return dis, nx

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(self.n_subset_ul)
        self.net.clf.cpu()
        self.net.clf.eval()
        dis = np.zeros(unlabeled_idxs.shape)
        attacked_images = []
        for i in tqdm(range(len(unlabeled_idxs)), ncols=100):
            x, y, idx = unlabeled_data[i]
            d, x_adv = self.cal_dis(x)
            dis[i] = d
            attacked_images.append(x_adv)
        self.net.clf.cuda()
        selected_idxs = dis.argsort()[:n]
        if add_adv:
            return unlabeled_idxs[selected_idxs], np.array(attacked_images)[selected_idxs]

        return unlabeled_idxs[selected_idxs], []

    def add_extra(self, extra_data):
        self.dataset.extra_data = np.concatenate(self.dataset.extra_data, extra_data)

    def attack_fn(self, X, **kwargs):
        """attack_fn to be implemented by child classes"""
        pass