# from torchvision import transforms
from handlers import MNIST_Handler, SVHN_Handler, CIFAR10_Handler
from data import get_MNIST, get_FashionMNIST, get_SVHN, get_CIFAR10
from nets import Net, MNIST_Net, SVHN_Net, CIFAR10_Net
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
    LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
    KMeansSampling, KCenterGreedy, BALDDropout, \
    AdversarialBIM, AdversarialPGD, AdversarialDeepFool

params = {'MNIST':
          {'n_epoch': 5,
           'train_args': {'batch_size': 64, 'num_workers': 0},
           'test_args': {'batch_size': 1000, 'num_workers': 0},
           'optimizer': 'rmsprop',
           'optimizer_args': {'lr': 0.005}},
          'FashionMNIST':
              {'n_epoch': 50,
               'train_args': {'batch_size': 64, 'num_workers': 0},
               'test_args': {'batch_size': 1000, 'num_workers': 0},
               'optimizer': 'rmsprop',
               'optimizer_args': {'lr': 0.005}},
          'SVHN':
              {'n_epoch': 20,
               'train_args': {'batch_size': 64, 'num_workers': 0},
               'test_args': {'batch_size': 1000, 'num_workers': 0},
               'optimizer': 'rmsprop',
               'optimizer_args': {'lr': 0.01, 'momentum': 0.5}},
          'CIFAR10':
              {'n_epoch': 20,
               'train_args': {'batch_size': 64, 'num_workers': 0},
               'test_args': {'batch_size': 1000, 'num_workers': 0},
               'optimizer': 'rmsprop',
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


def get_dataset(name, pool_size):
    if name == 'MNIST':
        return get_MNIST(get_handler(name), pool_size)
    elif name == 'FashionMNIST':
        return get_FashionMNIST(get_handler(name), pool_size)
    elif name == 'SVHN':
        return get_SVHN(get_handler(name), pool_size)
    elif name == 'CIFAR10':
        return get_CIFAR10(get_handler(name), pool_size)
    else:
        raise NotImplementedError


def get_net(name, device, reset=True):
#     if device == 'cpu':
#         params[name]['train_args']['num_workers'] = 0
    # print(params[name]['train_args']['num_workers'])
    if name == 'MNIST':
        return Net(MNIST_Net, params[name], device, reset)
    elif name == 'FashionMNIST':
        return Net(MNIST_Net, params[name], device, reset)
    elif name == 'SVHN':
        return Net(SVHN_Net, params[name], device, reset)
    elif name == 'CIFAR10':
        return Net(CIFAR10_Net, params[name], device, reset)
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
    elif name == "AdversarialPGD":
        strategy = AdversarialPGD
    else:
        raise NotImplementedError
    return strategy

# albl_list = [MarginSampling(X_tr, Y_tr, idxs_lb, net, handler, args),
#              KMeansSampling(X_tr, Y_tr, idxs_lb, net, handler, args)]
# strategy = ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, args, strategy_list=albl_list, delta=0.1)
def log_to_file(file_name, line):
    file = open(file_name, 'a')
    file.write(line)
    if not line.endswith('\n'):
        file.write('\n')
    file.close()
