# from torchvision import transforms
from handlers import MNIST_Handler, SVHN_Handler, CIFAR10_Handler
from data import get_MNIST, get_FashionMNIST, get_SVHN, get_CIFAR10
from nets import Net, MNIST_Net, SVHN_Net, CIFAR10_Net, oMNIST_Net, LeNet5
from query_strategies import RandomSampling, LeastConfidence, MarginSampling, EntropySampling, \
    LeastConfidenceDropout, MarginSamplingDropout, EntropySamplingDropout, \
    KMeansSampling, KCenterGreedy, BALDDropout, \
    AdversarialBIM, AdversarialPGD, AdversarialDeepFool
from resnet import ResNet18


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


def get_dataset(name, pool_size, n_adv_test):
    if name.lower() == 'mnist':
        return get_MNIST(get_handler(name), pool_size)
    elif name.lower() == 'fashionmnist':
        return get_FashionMNIST(get_handler(name), pool_size)
    elif name.lower() == 'svhn':
        return get_SVHN(get_handler(name), pool_size)
    elif name.lower() == 'cifar10':
        return get_CIFAR10(get_handler(name), pool_size, n_adv_test)
    else:
        raise NotImplementedError


def get_net(params, device):
    name = params['net_arch']
    if name.lower() == 'lenet5':
        return Net(LeNet5, params['name'], device, params['repeat'], params['reset'], params['advtrain_mode'])
    elif name.lower() == 'fashionmnist':
        return Net(MNIST_Net, params['name'], device, params['repeat'], params['reset'], params['advtrain_mode'])
    elif name.lower() == 'svhn':
        return Net(SVHN_Net, params['name'], device, params['repeat'], params['reset'], params['advtrain_mode'])
    elif name.lower() == 'cifar10':
        return Net(CIFAR10_Net, params, device)
    elif name.lower()=='resnet18':
        return Net(ResNet18, params, device)
    else:
        raise NotImplementedError


def get_strategy(name):
    if name.lower() == "randomsampling":
        strategy = RandomSampling
    elif name.lower() == "leastconfidence":
        strategy = LeastConfidence
    elif name.lower() == "marginsampling":
        strategy = MarginSampling
    elif name.lower() == "entropysampling":
        strategy = EntropySampling
    elif name.lower() == "leastconfidencedropout":
        strategy = LeastConfidenceDropout
    elif name.lower() == "marginsamplingdropout":
        strategy = MarginSamplingDropout
    elif name.lower() == "entropysamplingdropout":
        strategy = EntropySamplingDropout
    elif name.lower() == "kmeanssampling":
        strategy = KMeansSampling
    elif name.lower() == "kcentergreedy":
        strategy = KCenterGreedy
    elif name.lower() == "balddropout":
        strategy = BALDDropout
    elif name.lower() == "adversarialbim":
        strategy = AdversarialBIM
    elif name.lower() == "adversarialdeepfool":
        strategy = AdversarialDeepFool
    elif name.lower() == "adversarialpgd":
        strategy = AdversarialPGD
    else:
        raise NotImplementedError
    return strategy
