# import numpy as np
# import torch
from .strategy import Strategy
from tqdm import tqdm

class AdversarialBIM(AdversarialStrategy):
    def __init__(self, dataset, net,
                    n_subset_ul=None,
                    pseudo_labeling=True,
                    diversity=False, **kwargs):
        super().__init__(dataset, net,
                        n_subset_ul,
                        pseudo_labeling,
                        diversity, **kwargs)


    def attack_fn(self, X, **kwargs):
        """PGD attack"""
        return projected_gradient_descent(true_image, ,
                                            eps=0.3, eps_iter=1e-2, nb_iter=10, norm=np.inf,
                                            targeted=False, rand_init=True, rand_minmax=0.3, 
                                            sanity_checks=False)