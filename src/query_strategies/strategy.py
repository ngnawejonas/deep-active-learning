import numpy as np
import torch
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from data import Data
from nets import Net
from utils import compute_norm, get_attack_fn, clever_score  # , log_to_file


class Strategy:
    def __init__(self, dataset: Data, net: Net, pseudo_labeling=False, max_iter=None, dist_file_name=None, id_exp=0):
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
        dis_inf = compute_norm( x - x_adv.cpu(), np.inf)
        dis_2 = compute_norm( x - x_adv.cpu(), 2)
        dis = {'2': dis_2, 'inf': dis_inf}
       
        return nb_iter, dis, cumul_dis
    
    # def cal_robust_dis(self, x, min_pixel_value=0, max_pixel_value=1.):
    #     classifier = PyTorchClassifier(
    #     model=self.net.clf,
    #     clip_values=(min_pixel_value, max_pixel_value),
    #     loss=None,
    #     optimizer=None,
    #     input_shape=(1, 32, 32),
    #     nb_classes=10,
    #     )
    #     res1 = clever_u(classifier, x.numpy(), nb_batches=10, batch_size=1, radius=0.3, norm=np.inf, pool_factor=3)
    #     return res1

    def eval_test_dis(self):
        self.net.clf.eval()
        attack_name = self.net.params['dis_test_attack']['name']
        attack_params = self.net.params['dis_test_attack']['args'] if self.net.params['dis_test_attack'].get('args') else {}
        if attack_params.get('norm'):
            attack_params['norm'] = float(attack_params['norm'])
        data_loader = DataLoader(self.dataset.get_adv_test_data())

        dis_inf_list = [] #np.zeros(self.dataset.n_adv_test)
        dis_2_list = []  # np.zeros(self.dataset.n_adv_test)
        nb_iter_list = [] #np.zeros(self.dataset.n_adv_test)
        cumul_dis_inf_list = [] #np.zeros(self.dataset.n_adv_test)
        cumul_dis_2_list = [] # np.zeros(self.dataset.n_adv_test)
        clever_dis_list = []

        correct_idxs = []
        success_attack_idxs = []
        minpx, maxpx = self.dataset.get_min_max_pixel_values()
        clever_args = {'min_pixel_value': minpx, 'max_pixel_value': maxpx}
        # print(minpx, maxpx)
        i = 0
        for x, y, _ in tqdm(data_loader):
            initial_label = self.net.predict_example(x)
            if y == initial_label:
                correct_idxs.append(i)

            nb_iter, dis, cumul_dis = self.cal_dis_test(x, attack_name, **attack_params)

            if nb_iter < self.max_iter:
                success_attack_idxs.append(i)
            
            # clever score/dis
            # clever_dis = clever_score(self.net.clf, x[0], **clever_args)
           
            # clever_dis_list.append(clever_dis)
            dis_inf_list.append(dis['inf'])
            dis_2_list.append(dis['2'])
            nb_iter_list.append(nb_iter)
            cumul_dis_inf_list.append(cumul_dis['inf'])
            cumul_dis_2_list.append(cumul_dis['2'])
            
            i=i+1

        dis_list = {'d_inf': dis_inf_list,
                    'd_2': dis_2_list,
                    'cumul_inf': cumul_dis_inf_list,
                    'cumul_2': cumul_dis_2_list,}
                    # 'clever_dis': clever_dis_list}

        filter_idxs = {'initial_correct': correct_idxs, 'success': success_attack_idxs}
        return dis_list, nb_iter_list, filter_idxs
