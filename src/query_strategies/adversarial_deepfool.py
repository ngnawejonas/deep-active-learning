import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from attacks import deepfool_attack
from .strategy import Strategy
from .adversarial_strategy import AdversarialStrategy


class AdversarialDeepFool(AdversarialStrategy):
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

        self.attack_name = 'deepfool'
