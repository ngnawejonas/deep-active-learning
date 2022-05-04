# from torchvision import transforms
from handlers import MNIST_Handler, SVHN_Handler, CIFAR10_Handler
from data import get_MNIST, get_FashionMNIST, get_SVHN, get_CIFAR10
from nets import Net, MNIST_Net, SVHN_Net, CIFAR10_Net
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
    LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
    KMeansSampling, KCenterGreedy, BALDDropout, \
    AdversarialBIM, AdversarialDeepFool

params = {'MNIST':
          {'n_epoch': 10,
           'train_args': {'batch_size': 64, 'num_workers': 1},
           'test_args': {'batch_size': 1000, 'num_workers': 1},
           'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
          'FashionMNIST':
              {'n_epoch': 10,
               'train_args': {'batch_size': 64, 'num_workers': 1},
               'test_args': {'batch_size': 1000, 'num_workers': 1},
               'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
          'SVHN':
              {'n_epoch': 20,
               'train_args': {'batch_size': 64, 'num_workers': 1},
               'test_args': {'batch_size': 1000, 'num_workers': 1},
               'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
          'CIFAR10':
              {'n_epoch': 20,
               'train_args': {'batch_size': 64, 'num_workers': 1},
               'test_args': {'batch_size': 1000, 'num_workers': 1},
               'optimizer_args': {'lr': 0.05, 'momentum': 0.3}}
          }


def get_handler(name):
    if name == 'MNIST':
        return MNIST_Handler
    elif name == 'FashionMNIST':
        return MNIST_Handler
    elif name == 'SVHN':
        return SVHN_Handler
    elif name == 'CIFAR10':
        return CIFAR10_Handler
    else:
        raise NotImplementedError('Unhandled dataset')


def get_dataset(name):
    if name == 'MNIST':
        return get_MNIST(get_handler(name))
    elif name == 'FashionMNIST':
        return get_FashionMNIST(get_handler(name))
    elif name == 'SVHN':
        return get_SVHN(get_handler(name))
    elif name == 'CIFAR10':
        return get_CIFAR10(get_handler(name))
    else:
        raise NotImplementedError


def get_net(name, device):
    if name == 'MNIST':
        return Net(MNIST_Net, params[name], device)
    elif name == 'FashionMNIST':
        return Net(MNIST_Net, params[name], device)
    elif name == 'SVHN':
        return Net(SVHN_Net, params[name], device)
    elif name == 'CIFAR10':
        return Net(CIFAR10_Net, params[name], device)
    else:
        raise NotImplementedError


def get_params(name):
    return params[name]


def get_strategy(name):
    if name == "RandomSampling":
        strategy = RandomSampling
    elif name == "LeastConfidence":
        strategy = LeastConfidence
    elif name == "MarginSampling":
        strategy = MarginSampling
    elif name == "EntropySampling":
        strategy = EntropySampling
    elif name == "LeastConfidenceDropout":
        strategy = LeastConfidenceDropout
    elif name == "MarginSamplingDropout":
        strategy = MarginSamplingDropout
    elif name == "EntropySamplingDropout":
        strategy = EntropySamplingDropout
    elif name == "KMeansSampling":
        strategy = KMeansSampling
    elif name == "KCenterGreedy":
        strategy = KCenterGreedy
    elif name == "BALDDropout":
        strategy = BALDDropout
    elif name == "AdversarialBIM":
        strategy = AdversarialBIM
    elif name == "AdversarialDeepFool":
        strategy = AdversarialDeepFool
    else:
        raise NotImplementedError
    return strategy

# albl_list = [MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args),
#              KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)]
# strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
