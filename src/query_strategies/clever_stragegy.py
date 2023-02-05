import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import clever_score

from .strategy import Strategy


class CleverStrategy(Strategy):
    def __init__(self, dataset, net,
                 n_subset_ul=None, **kwargs):
        super().__init__(dataset, net, **kwargs)
        self.n_subset_ul = n_subset_ul  # number of unlabeled data to attack

    def check_querying(self, n_query):
        if self.n_subset_ul < n_query:
            raise ValueError(
                f"Impossible to query more than {self.n_subset_ul}. n_query = {n_query}!")

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(
            self.n_subset_ul)
        self.net.clf.eval()
        distances = np.zeros(unlabeled_idxs.shape)
        data_loader = DataLoader(unlabeled_data)
        minpx, maxpx = self.dataset.get_min_max_pixel_values()
        clever_args = {'min_pixel_value': minpx, 'max_pixel_value': maxpx}
        i = 0
        for x, _ , _ in tqdm(data_loader):
            distances[i] = clever_score(self.net.clf, x[0], **clever_args)
            i += 1
        ##
        return unlabeled_idxs[distances.argsort()[:n]]