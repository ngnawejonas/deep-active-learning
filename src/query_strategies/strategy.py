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

    def cal_dis_test(self, x, attack_name, **attack_params):
        attack_fn = get_attack_fn(attack_name)
        x_adv, nb_iter, cumul_dis_inf, cumul_dis_2 = attack_fn(self.net.clf, x.to(self.net.device), **attack_params)
        
        dis_inf = torch.linalg.norm(torch.ravel(x - x_adv.cpu()), ord=np.inf)
        dis_2 = torch.linalg.norm(x - x_adv.cpu())
        
        return nb_iter, dis_inf, dis_2, cumul_dis_inf, cumul_dis_2


    def eval_test_dis(self):
        self.net.clf.eval()
        attack_name = self.net.params['dis_test_attack']['name']
        attack_params = self.net.params['dis_test_attack']['args']
        if attack_params.get('norm'):
            attack_params['norm'] = np.inf if attack_params['norm']=='np.inf' else 2
        iter_loader = iter(DataLoader(self.dataset.get_adv_test_data()))

        dis_inf_list = np.zeros(self.dataset.n_adv_test)
        # dis_inf2_list = np.zeros(self.dataset.n_adv_test)
        dis_2_list = np.zeros(self.dataset.n_adv_test)
        nb_iter_list = np.zeros(self.dataset.n_adv_test)
        cumul_dis_inf_list = np.zeros(self.dataset.n_adv_test)
        cumul_dis_2_list = np.zeros(self.dataset.n_adv_test)

        correct_idxs = []
        for i in tqdm(range(self.dataset.n_adv_test), ncols=100):
            x, y, _ = iter_loader.next()
            initial_label = self.net.predict_example(x)
            if y == initial_label:
                correct_idxs.append(i)
            nb_iter, dis_inf, dis_2, cumul_dis_inf, cumul_dis_2 = self.cal_dis_test(x, attack_name, **attack_params)
            
            # log_to_file(self.dist_file_name, f'{self.id_exp}, {i}, {cumul_dis_2:.3f}, {cumul_dis_inf:.3f}, {nb_iter}')

            dis_inf_list[i] = dis_inf.detach().numpy()
            dis_2_list[i] = dis_2.detach().numpy()
            nb_iter_list[i] = nb_iter
            dis_inf_list[i] = cumul_dis_inf.detach().numpy()
            dis_2_list[i] = cumul_dis_2.detach().numpy()

        return dis_inf_list, dis_2_list, cumul_dis_inf_list, cumul_dis_2_list, nb_iter_list, correct_idxs
