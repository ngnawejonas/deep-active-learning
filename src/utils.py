import numpy as np
import torch
from attacks import test_pgd_attack, pgd_attack, deepfool_attack
from attacks import test_deepfool_attack, bim_attack, test_bim_attack  # , test_auto_attack
# from pgd_adaptive import projected_gradient_descent as pgd_attack

DMAX_INF = 1
DMAX_2 = 28

def compute_norm(x, norm):
    with torch.no_grad():
        if norm == np.inf:
            return torch.linalg.norm(torch.ravel(x.cpu()), ord=np.inf).numpy()
        elif norm == 2:
            return torch.linalg.norm(x.cpu()).numpy()
        else:
            raise NotImplementedError

def log_to_file(file_name, line):
    filepath = 'results/'+file_name
    file = open(filepath, 'a')
    file.write(line)
    if not line.endswith('\n'):
        file.write('\n')
    file.close()


def get_attack_fn(name=None, for_dis_cal=False):
    if not for_dis_cal:
        # if name == 'autoattack':
        #     return test_auto_attack
        if name == 'bim':
            return test_bim_attack
        elif name == 'pgd':
            return test_pgd_attack
        elif name == 'deepfool':
            return test_deepfool_attack
        else:
            raise NotImplementedError(
                'Attack "{}" not implemented'.format(name))
    else:
        if name == 'bim':
            return bim_attack
        elif name == 'pgd':
            return pgd_attack
        elif name == 'deepfool':
            return deepfool_attack
        else:
            raise NotImplementedError(
                'Attack "{}" not implemented'.format(name))
