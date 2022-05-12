import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy
from tqdm import tqdm

from .adversarial_strategy import AdversarialStrategy

class AdversarialDeepFool(AdversarialStrategy):
    def __init__(self, dataset, net,
                    repeat= 1,
                    pseudo_labeling=True,
                    max_iter=10,
                    n_subset_ul=None,
                    diversity=False,
                    dist_file_name=None, **kwargs):
        super().__init__(dataset, net,
                        repeat,
                        pseudo_labeling,
                        max_iter,
                        n_subset_ul,
                        diversity, dist_file_name, **kwargs)

    def attack_fn(self, x):
        """DeepFool attack"""
        nx =  x.clone()
        nx.requires_grad_()
        out = self.net.clf(nx)
        n_class = out.shape[1]
        py = out.max(1)[1].item()
        out[0, py].backward(retain_graph=True)
        grad_np = nx.grad.data.clone()
        value_l = np.inf
        ri = None
        for i in range(n_class):
            if i == py:
                continue
            nx.grad.data.zero_()
            out[0, i].backward(retain_graph=True)
            grad_i = nx.grad.data.clone()

            wi = grad_i - grad_np
            fi = out[0, i] - out[0, py]
            value_i = torch.abs(fi) / torch.norm(wi.flatten())

            if value_i < value_l:
                ri = value_i/torch.norm(wi.flatten()) * wi

        return x + ri

