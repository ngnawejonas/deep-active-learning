import numpy as np
from attacks import test_pgd_attack, pgd_attack, deepfool_attack
from attacks import test_deepfool_attack, bim_attack, test_bim_attack  # , test_auto_attack
# from pgd_adaptive import projected_gradient_descent as pgd_attack


def log_to_file(file_name, line):
    filepath = 'results/'+file_name
    file = open(filepath, 'a')
    file.write(line)
    if not line.endswith('\n'):
        file.write('\n')
    file.close()


def get_attack_fn(name='pgd', for_dis_cal=False):
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
        elif name == 'test_pgd':
            return test_pgd_attack
        elif name == 'deepfool':
            return deepfool_attack
        else:
            raise NotImplementedError(
                'Attack "{}" not implemented'.format(name))
