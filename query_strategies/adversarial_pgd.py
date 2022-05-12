import numpy as np
# import torch
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method        
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

from .adversarial_strategy import AdversarialStrategy

class AdversarialPGD(AdversarialStrategy):
    def __init__(self, dataset, net,
                    pseudo_labeling=True,
                    max_iter=10,
                    n_subset_ul=None,
                    diversity=False,
                    dist_file_name=None,
                    id_exp=0, **kwargs):
        super().__init__(dataset, net,
                        pseudo_labeling,
                        max_iter,
                        n_subset_ul,
                        diversity, dist_file_name, id_exp, **kwargs)


    def attack_fn(self, X):
        """PGD attack"""
        return projected_gradient_descent(self.net.clf, X, **self.params)