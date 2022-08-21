import numpy as np
import torch.optim as optim
from attacks import pgd_attack, bim_attack, fgsm_attack, deepfool_attack
# from torchvision import transforms
from handlers import MNIST_Handler, SVHN_Handler, CIFAR10_Handler
from data import get_MNIST, get_FashionMNIST, get_SVHN, get_CIFAR10
from nets import Net, MNIST_Net, SVHN_Net, CIFAR10_Net, oMNIST_Net
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
    LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
    KMeansSampling, KCenterGreedy, BALDDropout, \
    AdversarialBIM, AdversarialPGD, AdversarialDeepFool



def get_handler(name):
    if name.lower() == 'mnist':
        return MNIST_Handler
    elif name.lower() == 'fashionmnist':
        return MNIST_Handler
    elif name.lower() == 'svhn':
        return SVHN_Handler
    elif name.lower() == 'cifar10':
        return CIFAR10_Handler
    else:
        raise NotImplementedError('Unhandled dataset')


def get_dataset(name, pool_size):
    if name.lower() == 'mnist':
        return get_MNIST(get_handler(name), pool_size)
    elif name.lower() == 'fashionmnist':
        return get_FashionMNIST(get_handler(name), pool_size)
    elif name.lower() == 'svhn':
        return get_SVHN(get_handler(name), pool_size)
    elif name.lower() == 'cifar10':
        return get_CIFAR10(get_handler(name), pool_size)
    else:
        raise NotImplementedError


def get_net(params, device):
    name = params['net_arch']
    if name.lower() == 'MNIST':
        return Net(oMNIST_Net, params['name'], device, params['repeat'], params['reset'], params['advtrain_mode'])
    elif name.lower() == 'FashionMNIST':
        return Net(MNIST_Net, params['name'], device, params['repeat'], params['reset'], params['advtrain_mode'])
    elif name.lower() == 'SVHN':
        return Net(SVHN_Net, params['name'], device, params['repeat'], params['reset'], params['advtrain_mode'])
    elif name.lower() == 'CIFAR10':
        return Net(CIFAR10_Net, params['name'], device, params['repeat'], params['reset'], params['advtrain_mode'])
    else:
        raise NotImplementedError


def get_strategy(name):
    if name.lower() == "RandomSampling":
        strategy = RandomSampling
    elif name.lower() == "LeastConfidence":
        strategy = LeastConfidence
    elif name.lower() == "MarginSampling":
        strategy = MarginSampling
    elif name.lower() == "EntropySampling":
        strategy = EntropySampling
    elif name.lower() == "LeastConfidenceDropout":
        strategy = LeastConfidenceDropout
    elif name.lower() == "MarginSamplingDropout":
        strategy = MarginSamplingDropout
    elif name.lower() == "EntropySamplingDropout":
        strategy = EntropySamplingDropout
    elif name.lower() == "KMeansSampling":
        strategy = KMeansSampling
    elif name.lower() == "KCenterGreedy":
        strategy = KCenterGreedy
    elif name.lower() == "BALDDropout":
        strategy = BALDDropout
    elif name.lower() == "AdversarialBIM":
        strategy = AdversarialBIM
    elif name.lower() == "AdversarialDeepFool":
        strategy = AdversarialDeepFool
    elif name.lower() == "AdversarialPGD":
        strategy = AdversarialPGD
    else:
        raise NotImplementedError
    return strategy

# albl_list = [MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args),
#              KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)]
# strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
def log_to_file(file_name, line):
    filepath = 'results/'+file_name
    file = open(filepath, 'a')
    file.write(line)
    if not line.endswith('\n'):
        file.write('\n')
    file.close()