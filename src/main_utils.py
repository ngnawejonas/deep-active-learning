# from torchvision import transforms
from handlers import MNIST_Handler, SVHN_Handler, CIFAR10_Handler
from data import get_MNIST, get_FashionMNIST, get_SVHN, get_CIFAR10, get_binary_MNIST
from nets import Net, SVHN_Net, CIFAR10_Net, BinaryLeNet5, oMNIST_Net, LeNet5
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
    if name.lower() == 'binary_mnist':
        return get_binary_MNIST(get_handler('mnist'), pool_size, n_adv_test)
    if name.lower() == 'mnist':
        return get_MNIST(get_handler(name), pool_size, n_adv_test)
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
    if name.lower() == 'binarylenet5':
        return Net(BinaryLeNet5, params, device)
    if name.lower() == 'lenet5':
        return Net(LeNet5, params, device)
    # elif name.lower() == 'fashionmnist':
    #     return Net(MNIST_Net, params, device)
    # elif name.lower() == 'svhn':
    #     return Net(MNIST_Net, params, device)
    # elif name.lower() == 'cifar10':
    #     return Net(CIFAR10_Net, params, device)
    elif name.lower() == 'resnet18':
        return Net(ResNet18, params, device)
    else:
        raise NotImplementedError


def get_strategy(name):
    if name.lower() == "random":
        strategy = RandomSampling
    elif name.lower() == "leastconfidence":
        strategy = LeastConfidence
    elif name.lower() == "margin":
        strategy = MarginSampling
    elif name.lower() == "entropy":
        strategy = EntropySampling
    elif name.lower() == "leastconfidencedropout":
        strategy = LeastConfidenceDropout
    elif name.lower() == "marginsamplingdropout":
        strategy = MarginSamplingDropout
    elif name.lower() == "entropysamplingdropout":
        strategy = EntropySamplingDropout
    elif name.lower() == "kmeans":
        strategy = KMeansSampling
    elif name.lower() == "coreset":
        strategy = KCenterGreedy
    elif name.lower() == "bald":
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
