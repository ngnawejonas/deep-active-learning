import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from train_utils import adv_params, get_attack_fn, log_to_file

class Strategy:
    def __init__(self, dataset, net, pseudo_labeling=False, max_iter=10, dist_file_name=None, id_exp=0):
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

    def eval_adv_acc(self):
        preds = self.predict_adv(self.dataset.get_adv_test_data())
        acc = self.dataset.cal_adv_test_acc(preds)
        return acc

    def cal_dis_test(self, x, attack_fn, **attack_params):
        x_i = x.clone()
        initial_label = self.net.predict_example(x_i)
        i_iter = 0
        while self.net.predict_example(x_i) == initial_label and i_iter < self.max_iter:
            x_i = attack_fn(self.net.clf, x_i.to(self.net.device), **attack_params)
            i_iter += 1
        x_i = x_i.cpu()
        dis = torch.norm(x_i - x)
        return dis.detach(), x_i.detach().squeeze(0)


    def eval_dis(self):
        self.net.clf.eval()
        attack_name = adv_params['test_attack']['name']
        attack_params = adv_params['test_attack']['args']
        attack_fn = get_attack_fn(attack_name)
        iter_loader = iter(DataLoader(self.dataset.get_adv_test_data()))
        for i in tqdm(range(self.dataset.n_adv_test), ncols=100):
            x, y, _ = iter_loader.next()
            dis, x_adv = self.cal_dis_test(x, attack_fn, **attack_params)
            log_to_file(self.dist_file_name, f'{self.id_exp}, {i}, {dis.numpy()}')