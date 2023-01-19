import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import compute_norm, get_attack_fn, log_to_file

from .strategy import Strategy


class AdversarialStrategy(Strategy):
    def __init__(self, dataset, net,
                 pseudo_labeling=True,
                 max_iter=None,
                 n_subset_ul=None,
                 diversity=False,
                 dist_file_name=None,
                 id_exp=0, cumul=False, norm=None, **kwargs):
        super().__init__(dataset, net, pseudo_labeling, max_iter, dist_file_name, id_exp)
        self.diversity = diversity
        self.n_subset_ul = n_subset_ul  # number of unlabeled data to attack
        self.cumul = cumul
        self.norm = float(norm)
        self.attack_params = kwargs['args'] if kwargs.get('args') else {}
        if self.attack_params.get('norm'):
            self.attack_params['norm'] = float(self.attack_params['norm'])
            assert self.norm == self.attack_params['norm']

        self.adv_dist_file_name = "train_"+dist_file_name
        self.attack_name = None

    def check_querying(self, n_query):
        if self.n_subset_ul < n_query:
            raise ValueError(
                f"Impossible to query more than {self.n_subset_ul}. n_query = {n_query}!")

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(
            self.n_subset_ul)
        self.net.clf.eval()
        distances = np.zeros(unlabeled_idxs.shape)
        adv_images = []
        data_loader = DataLoader(unlabeled_data)
        attack_fn = get_attack_fn(self.attack_name, for_dis_cal=True)
        i = 0
        for x, y, _ in tqdm(data_loader):
            x_adv, nb_iter, cumul_dis = attack_fn(self.net.clf, x.to(
                self.net.device), self.max_iter, **self.attack_params)

            if self.cumul: # to be debugged
                if torch.is_tensor(cumul_dis):
                    cumul_dis = cumul_dis.detach().numpy()
                distances[i] = cumul_dis
            else:
                dis = compute_norm(x - x_adv.cpu(), self.norm)
                distances[i] = dis.numpy()

            # log_to_file(self.adv_dist_file_name, f'{self.id_exp}, {i}, {dis:.3f}, {nb_iter}')
            adv_images.append(x_adv.squeeze(0).detach().cpu() if x.shape[0] == 1 else x_adv)
            i += 1
        ##
        if self.diversity:
            selected_idxs = self.f_diversity(distances, adv_images)
        else:
            selected_idxs = distances.argsort()[:n]

        # breakpoint()
        extra_data = None
        if self.pseudo_labeling and len(adv_images) > 0:
            extra_data = torch.stack(adv_images)[selected_idxs]
            if extra_data.shape[1:] == (3,32,32): # for cifar 10 extras
                extra_data = torch.reshape(extra_data, (-1,32,32,3))
        return unlabeled_idxs[selected_idxs], extra_data

    def f_diversity(self, distances, adv_images):
        print('diversity selection')
        perturbations = torch.Tensor(distances)
        index_perturbation = perturbations.argsort()
        adv = torch.stack(adv_images)
        sortedAdv = adv[index_perturbation]

        dist = []
        for i in range(self.n_subset_ul):
            for j in range(self.n_subset_ul):
                adv_dist = torch.norm(sortedAdv[i]-sortedAdv[j])
                dist.append(adv_dist.cpu().numpy())

        median_dist = np.median(np.unique(dist))
        selected_idxs = []

        for i in range(self.n_subset_ul):
            index_max = np.argmax(
                dist[self.n_subset_ul*i:self.n_subset_ul*(i+1)])
            max_dist = dist[(self.n_subset_ul*i)+index_max]
            if max_dist > median_dist:
                selected_idxs.append(index_perturbation[i])
        return selected_idxs
