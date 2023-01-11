import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import get_attack_fn  # , log_to_file


class Strategy:
    def __init__(self, dataset, net, pseudo_labeling=False, max_iter=None, dist_file_name=None, id_exp=0):
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

    def eval_acc_on_adv_test_data(self):
        preds = self.predict(self.dataset.get_adv_test_data())
        acc = self.dataset.cal_adv_test_acc(preds)
        return acc

    def eval_adv_acc(self):
        preds = self.predict_adv(self.dataset.get_adv_test_data())
        acc = self.dataset.cal_adv_test_acc(preds)
        return acc

    def cal_dis_test(self, x, attack_name, **attack_params):
        attack_fn = get_attack_fn(attack_name, for_dis_cal=True)
        x_adv, nb_iter, cumul_dis = attack_fn(self.net.clf, x.to(self.net.device), self.max_iter, **attack_params)
        # breakpoint()
        if nb_iter < self.max_iter:
            eta = x - x_adv.cpu()
            dis = compute_norm(eta, attack_params['norm'])
        else:
            dis = np.inf

        if torch.is_tensor(cumul_dis):
            cumul_dis = cumul_dis.detach().numpy()
       
        return nb_iter, dis, cumul_dis

    def eval_test_dis(self):
        self.net.clf.eval()
        attack_name = self.net.params['dis_test_attack']['name']
        attack_params = self.net.params['dis_test_attack']['args'] if self.net.params['dis_test_attack'].get('args') else {}
        if attack_params.get('norm'):
            attack_params['norm'] = float(attack_params['norm'])
        data_loader = DataLoader(self.dataset.get_adv_test_data())

        dis_list = [] #np.zeros(self.dataset.n_adv_test)
        nb_iter_list = [] #np.zeros(self.dataset.n_adv_test)
        cumul_dis_list = [] #np.zeros(self.dataset.n_adv_test)

        correct_idxs = []
        i = 0
        for x, y, _ in tqdm(data_loader):
            initial_label = self.net.predict_example(x)
            if y == initial_label:
                correct_idxs.append(i)
            nb_iter, dis, cumul_dis = self.cal_dis_test(x, attack_name, **attack_params)

            dis_list.append(dis)
            nb_iter_list.append(nb_iter)
            cumul_dis_list.append(cumul_dis)
            i=i+1

        return dis_list, cumul_dis_list, nb_iter_list, correct_idxs


def compute_norm(x, norm):
    if norm == np.inf:
        return torch.linalg.norm(torch.ravel(x), ord=np.inf).detach().numpy()
    elif norm == 2:
        return torch.linalg.norm(x).detach().numpy()
    else:
        raise NotImplementedError
