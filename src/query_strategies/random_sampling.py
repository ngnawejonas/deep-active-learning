import numpy as np
from .strategy import Strategy


class RandomSampling(Strategy):
    def __init__(self, dataset, net, **kwargs):
        super().__init__(dataset, net, **kwargs)

    def query(self, n):
        return np.random.choice(np.where(self.dataset.labeled_idxs == 0)[0], n, replace=False)
