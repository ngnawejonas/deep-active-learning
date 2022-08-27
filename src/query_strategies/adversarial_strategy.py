import numpy as np
import torch
from .strategy import Strategy
from tqdm import tqdm
from torch.utils.data import DataLoader

from train_utils import log_to_file, get_attack_fn

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
        self.attack_params = kwargs
        self.adv_dist_file_name = "train_"+dist_file_name
        self.attack_name = None

    def check_querying(self, n_query):
        if self.n_subset_ul < n_query:
            raise ValueError(f"Impossible to query more than {self.n_subset_ul}. n_query = {n_query}!")

    def query(self, n):
        unlabeled_idxs, unlabeled_data = self.dataset.get_unlabeled_data(self.n_subset_ul) 
        self.net.clf.eval()
        attack_fn = get_attack_fn(self.attack_name)
        distances = np.zeros(unlabeled_idxs.shape)
        adv_images = []
        iter_loader = iter(DataLoader(unlabeled_data))
        for i in tqdm(range(len(unlabeled_idxs)), ncols=100):
            x, y, _ = iter_loader.next()
            nb_iter, x_adv = self.cal_dis(x, attack_fn, **self.attack_params)
            dis_inf = torch.linalg.norm(torch.ravel(x - x_adv), ord=np.inf).detach()
            # dis_2 = torch.linalg.norm(x - x_adv)
            distances[i] = dis_inf
            log_to_file(self.adv_dist_file_name, f'{self.id_exp}, {i}, {dis_inf.numpy():.3f}, {nb_iter}')
            adv_images.append(x_adv.squeeze(0) if x.shape[0]==1 else x_adv)

        ##
        if self.diversity:
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
                index_max = np.argmax(dist[self.n_subset_ul*i:self.n_subset_ul*(i+1)])
                max_dist = dist[(self.n_subset_ul*i)+index_max]
                if max_dist > median_dist:
                    selected_idxs.append(index_perturbation[i])
            selected_idxs = selected_idxs[:n]
        ##
        else:
            selected_idxs = distances.argsort()[:n]

        # breakpoint()
        extra_data = None
        if self.pseudo_labeling:
            if len(adv_images)>0:
                extra_data = torch.stack(adv_images)[selected_idxs]

        return unlabeled_idxs[selected_idxs], extra_data
