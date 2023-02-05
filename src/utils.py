import numpy as np
import torch
from attacks import test_pgd_attack, pgd_attack, deepfool_attack
from attacks import test_deepfool_attack, bim_attack, test_bim_attack  # , test_auto_attack
# from pgd_adaptive import projected_gradient_descent as pgd_attack
from art.estimators.classification.pytorch import PyTorchClassifier
from art.metrics.metrics import empirical_robustness, clever_t, clever_u

MIN_PIXEL_VALUE = -3
MAX_PIXEL_VALUE = 3

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


def clever_score(model, x, **args):
    classifier = PyTorchClassifier(
    model=model,
    clip_values=(args['min_pixel_value'], args['max_pixel_value']),
    loss=None,
    optimizer=None,
    input_shape=(1, 32, 32),
    nb_classes=10,
    )
    res = clever_u(classifier, x.numpy(), 
                    nb_batches=10, 
                    batch_size=1, 
                    radius=0.3, #args['radius'], 
                    norm=np.inf, #args['norm'], 
                    pool_factor=3, verbose=False)
    return res