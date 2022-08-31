import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import get_attack_fn, log_to_file
# from pgd_adaptive import projected_gradient_descent

class Strategy:
    def __init__(self, dataset, net, pseudo_labeling=False, max_iter=100, dist_file_name=None, id_exp=0):
        self.dataset = dataset
        self.net = net
        self.pseudo_labeling = pseudo_labeling
        self.max_iter = max_iter
        self.dist_file_name = dist_file_name
        self.id_exp = id_exp

    def query(self, n):
        pass

    def update(self, pos_idxs, extra_data=None):
        self.dataset.labeled_idxs[pos_idxs] = True
        if extra_data is not None and self.pseudo_labeling:
            self.dataset.add_extra_data(pos_idxs, extra_data)

    def train(self):
        labeled_idxs, labeled_data = self.dataset.get_labeled_data()
        self.net.train(labeled_data)

    def predict(self, data):
        preds = self.net.predict(data)
        return preds

    def predict_adv(self, data):
        preds = self.net.predict_adv(data)
        return preds

    def predict_prob(self, data):
        probs = self.net.predict_prob(data)
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout(data, n_drop=n_drop)
        return probs

    def predict_prob_dropout_split(self, data, n_drop=10):
        probs = self.net.predict_prob_dropout_split(data, n_drop=n_drop)
        return probs
    
    def get_embeddings(self, data):
        embeddings = self.net.get_embeddings(data)
        return embeddings

    def eval_acc(self):
        preds = self.predict(self.dataset.get_test_data())
        acc = self.dataset.cal_test_acc(preds)
        return acc

    def eval_acc2(self):
        preds = self.predict(self.dataset.get_adv_test_data())
        acc = self.dataset.cal_adv_test_acc(preds)
        return acc

    def eval_adv_acc(self):
        preds = self.predict_adv(self.dataset.get_adv_test_data())
        acc = self.dataset.cal_adv_test_acc(preds)
        return acc

    def cal_dis(self, x, attack_name, **attack_params):
        # if attack_name.lower() == 'pgd':
        #     return projected_gradient_descent(self.net.clf, x.to(self.net.device), **attack_params)
        attack_fn = get_attack_fn(attack_name)
        x_i = x.clone()
        initial_label = self.net.predict_example(x_i)
        i_iter = 0
        while self.net.predict_example(x_i) == initial_label and i_iter < self.max_iter:
            x_i = attack_fn(self.net.clf, x_i.to(self.net.device), **attack_params)
            i_iter += 1
        x_i = x_i.cpu()
        return i_iter, x_i.detach().squeeze(0)


    def eval_test_dis(self):
        self.net.clf.eval()
        attack_name = self.net.params['test_attack']['name']
        attack_params = self.net.params['test_attack']['args']
        iter_loader = iter(DataLoader(self.dataset.get_adv_test_data()))

        dis_inf_list = np.zeros(self.dataset.n_adv_test)
        # dis_inf2_list = np.zeros(self.dataset.n_adv_test)
        dis_2_list = np.zeros(self.dataset.n_adv_test)
        nb_iter_list = np.zeros(self.dataset.n_adv_test)

        for i in tqdm(range(self.dataset.n_adv_test), ncols=100):
            x, y, _ = iter_loader.next()
            nb_iter, x_adv = self.cal_dis(x, attack_name, **attack_params)

            dis_inf = torch.linalg.norm(torch.ravel(x - x_adv), ord=np.inf)
            dis_2 = torch.linalg.norm(x - x_adv)

            dis_inf_list[i] = dis_inf.detach().numpy()
            dis_2_list[i] = dis_2.detach().numpy()

            nb_iter_list[i] = nb_iter

            log_to_file(self.dist_file_name, f'{self.id_exp}, {i}, {dis_2.numpy():.3f}, {dis_inf.numpy():.3f}, {nb_iter}')
        return dis_inf_list, dis_2_list, nb_iter_list
