import numpy as np
import torch
from .strategy import Strategy
from tqdm import tqdm
from torch.utils.data import DataLoader

from train_utils import log_to_file

class AdversarialStrategy(Strategy):
    def __init__(self, dataset, net,
                    pseudo_labeling=True,
                    max_iter=10,
                    n_subset_ul=None,
                    diversity=False,
                    dist_file_name=None,
                    id_exp=0, **kwargs):
        super().__init__(dataset, net, pseudo_labeling, max_iter, dist_file_name, id_exp)
        self.diversity = diversity
        self.n_subset_ul = n_subset_ul # number of unlabeled data to attack
        self.params = kwargs
        # self.dist_file_name = dist_file_name
        # self.id_exp = id_exp

    def check_querying(self, n_query):
        if self.n_subset_ul < n_query:
            raise ValueError(f"Impossible to query more than {self.n_subset_ul}. n_query = {n_query}!")

    # def cal_dis(self, x):
    #     x_i = x.clone()
    #     initial_label = self.net.predict_example(x_i)
    #     i_iter = 0
    #     while self.net.predict_example(x_i) == initial_label and i_iter < self.max_iter:
    #         x_i = self.attack_fn(x_i.to(self.net.device))
    #         i_iter += 1
    #     x_i = x_i.cpu()
    #     dis = torch.norm(x_i - x)
    #     return dis.detach(), x_i.detach().squeeze(0)

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(self.n_subset_ul) 
        self.net.clf.eval()
        distances = np.zeros(unlabeled_idxs.shape)
        adv_images = []
        iter_loader = iter(DataLoader(unlabeled_data))
        for i in tqdm(range(len(unlabeled_idxs)), ncols=100):
            x, y, _ = iter_loader.next()
            dis, x_adv = self.cal_dis(x)
            log_to_file(self.dist_file_name, f'{self.id_exp}, {i}, {np.round(dis.numpy(), 3)}')
            distances[i] = dis
            adv_images.append(x_adv.squeeze(0) if x.shape[0]==1 else x_adv)
        selected_idxs = distances.argsort()[:n]
        extra_data = None
        if self.pseudo_labeling:
            if len(adv_images)>0:
                extra_data = torch.stack(adv_images)[selected_idxs]

        return unlabeled_idxs[selected_idxs], extra_data

    def attack_fn(self, x):
        """attack_fn to be implemented by child classes"""
        pass