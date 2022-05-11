import numpy as np
# import torch
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method        
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent

from .adversarial_strategy import AdversarialStrategy

class AdversarialPGD(AdversarialStrategy):
    def __init__(self, dataset, net,
                    repeat= 1,
                    pseudo_labeling=True,
                    n_subset_ul=None,
                    diversity=False, **kwargs):
        super().__init__(dataset, net,
                        repeat,
                        pseudo_labeling,
                        n_subset_ul,
                        diversity, **kwargs)


    def attack_fn(self, X):
        """PGD attack"""
        return projected_gradient_descent(self.net.clf, X, **self.params)