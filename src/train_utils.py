import numpy as np
import torch.optim as optim
from attacks import pgd_attack, bim_attack, fgsm_attack, deepfool_attack


adv_params = {'train_attack':
                    {'name': 'pgd',
                      'args':{'eps': 0.3,
                                'eps_iter': 0.01,
                                'nb_iter': 10,
                                'norm': np.inf,
                                'targeted': False,
                                'rand_init': True,
                                'rand_minmax': 0.3
                              },
                       }, 
            'test_attack':
                {'name': 'pgd',
                  'args':{'eps': 0.3,
                            'eps_iter': 0.01,
                            'nb_iter': 10,
                            'norm': np.inf,
                            'targeted': False,
                            'rand_init': True,
                            'rand_minmax': 0.3
                          },
                   },
             }


def log_to_file(file_name, line):
    filepath = 'results/'+file_name
    file = open(filepath, 'a')
    file.write(line)
    if not line.endswith('\n'):
        file.write('\n')
    file.close()


def get_attack_fn(name='fgsm'):
    if name == 'fgsm':
        return fgsm_attack
    elif name == 'bim':
        return bim_attack
    elif name == 'pgd':
        return pgd_attack
    elif name == 'deepfool':
        return deepfool_attack
    else:
        raise NotImplementedError('Attack "{}" not implemented'.format(name))