from pprint import pprint
import os
import argparse
import time
import yaml
import random
import numpy as np
import torch
# import tensorflow as tf
from utils import get_dataset, get_net, get_strategy, log_to_file


if __name__ == "__main__":

    # os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    parser = argparse.ArgumentParser()

    parser.add_argument('--id_exp', type=int, default=0,
        help='id number of experiment')
    parser.add_argument('-cfg', dest="cfg", action='store_true')
    parser.set_defaults(cfg=False)

    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument(
        '--n_init_labeled',
        type=int,
        default=100,
        help="number of init labeled samples")
    parser.add_argument('--n_query', type=int, default=100,
                        help="number of queries per round")
    # parser.add_argument(
    #     '--n_test_adv',
    #     type=int,
    #     default=300,
    #     help="number of rounds")
    parser.add_argument(
        '--n_final_labeled',
        type=int,
        default=600,
        help="number of final labeled samples")
    parser.add_argument(
        '--dataset_name',
        type=str,
        default="MNIST",
        choices=[
            "MNIST",
            "FashionMNIST",
            "SVHN",
            "CIFAR10"],
        help="dataset")
    parser.add_argument(
        '--pool_size',
        type=int,
        default=50000,
        help="pool size, subset of the dataset to consider as pool.")

    # parser.add_argument(
    #     '--net_architect',
    #     type=str,
    #     default="None",
    #     choices=[
    #         "None",
    #         "resnet18",
    #         "vgg16"],
    #     help="network architecture")
    parser.add_argument('--repeat', type=int, default=0,
                        help="number of queries per round")
    parser.add_argument('--strategy_name', type=str, default="RandomSampling",
                        choices=["RandomSampling",
                                 "LeastConfidence",
                                 "MarginSampling",
                                 "EntropySampling",
                                 "LeastConfidenceDropout",
                                 "MarginSamplingDropout",
                                 "EntropySamplingDropout",
                                 "KMeansSampling",
                                 "KCenterGreedy",
                                 "BALDDropout",
                                 "AdversarialBIM",
                                 "AdversarialPGD",
                                 "AdversarialDeepFool"], help="query strategy")

    parser.add_argument('--reset', dest="reset", action='store_true')
    parser.add_argument('--no_reset', dest="reset", action='store_false')
    parser.set_defaults(reset=True)

    parser.add_argument('--nops', dest="pseudo_labeling", action='store_false')
    parser.set_defaults(pseudo_labeling=True)

    parser.add_argument('--advtrain', dest="adv_train_mode", action='store_true')
    parser.set_defaults(adv_train_mode=False)

    # parser.add_argument('--with_diversity', dest="diversity", action='store_true')
    # parser.set_defaults(diversity=False)    


    # parser.add_argument('--early_stopping', dest="early_stopping", action='store_true')
    # parser.add_argument('--no_early_stopping', dest="early_stopping", action='store_false')
    # parser.set_defaults(early_stopping=False)

    args = parser.parse_args()

    if args.cfg:
        try:
            with open('config.yaml', 'r') as config_file:
                config = yaml.load(config_file, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            print("Error in configuration file:", exc)
        pprint(config)

        id_exp = config['id_exp']
        seed = config['seed']
        n_init_labeled = config['n_init_labeled']
        n_query = config['n_query']
        n_final_labeled = config['n_final_labeled']
        # if n_final_labeled:
        #     n_round = (n_final_labeled - n_init_labeled) // n_query
        # else:
        #     n_round = config['n_round']

        dataset_name = config['dataset_name']
        pool_size = config['pool_size']
        strategy_name = config['strategy_name']
        reset = config['reset']
        repeat = config['repeat']
        # early_stopping = config['early_stopping']

    else:
        pprint(vars(args))
        id_exp = args.id_exp
        seed = args.seed
        n_init_labeled = args.n_init_labeled
        n_query = args.n_query
        n_final_labeled = args.n_final_labeled
        # if n_final_labeled:
        #     n_round = (n_final_labeled - n_init_labeled) // n_query
        # else:
        #     n_round = args.n_round
        # n_round = args.n_round
        dataset_name = args.dataset_name
        pool_size = args.pool_size
        strategy_name = args.strategy_name
        reset = args.reset
        repeat = args.repeat
        # early_stopping = config['early_stopping']
        adv_train_mode = args.adv_train_mode
    print()
    #
    # final = n_final_labeled if n_final_labeled else n_round*n_query+n_init_labeled
    strategy_name_on_file = strategy_name+"AdvTrain" if adv_train_mode else strategy_name
    ACC_FILENAME = '{}_{}_{}_{}_{}.txt'.format(
        strategy_name_on_file, dataset_name, 'resnet18', n_final_labeled, 'r'+str(repeat))
    #
    resultsDirName = 'results'
    try:
        os.mkdir(resultsDirName)
        print("Results directory ", resultsDirName ,  " Created ") 
    except FileExistsError:
        print("Results directory " , resultsDirName ,  " already exists")
    try:
        with open('./strategy_config.yaml', 'r') as config_file:
            strategy_config = yaml.load(config_file, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        print("Error in configuration file:", exc)

    # fix random seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using GPU: {use_cuda}')
    # print('getting dataset...')
    dataset = get_dataset(dataset_name, pool_size)        # load dataset
    # print('dataset loaded')
    net = get_net(dataset_name, device, repeat, reset, adv_train_mode)           # load network
    params = {}
    if strategy_config.get(strategy_name):
        params = strategy_config[strategy_name]
        if params.get('norm') and params.get('norm') == 'np.inf':
            params['norm'] = np.inf
        params['pseudo_labeling'] = args.pseudo_labeling
    params['dist_file_name'] = 'dist_'+ACC_FILENAME
    params['id_exp'] = id_exp
    pprint(params)

    strategy = get_strategy(strategy_name)(dataset, net, **params)       # load strategy

    if hasattr(strategy, 'n_subset_ul'):
        strategy.check_querying(n_query)

    # start experiment
    dataset.initialize_labels(n_init_labeled)
    print(f"size of labeled pool: {n_init_labeled}")
    print(f"size of unlabeled pool: {dataset.n_pool-n_init_labeled}")
    print(f"size of testing pool: {dataset.n_test}")
    print()
    # exit()
    start = time.time()
    # round 0 accuracy
    print("Round 0")
    t = time.time()
    strategy.train()
    print("train time: {:.2f} s".format(time.time() - t))
    print('testing...')
    acc = strategy.eval_acc()
    adv_acc = strategy.eval_adv_acc()
    strategy.eval_dis()

    print(f"Round 0 testing accuracy: {acc}")
    n_labeled = strategy.dataset.n_labeled()
    log_to_file(ACC_FILENAME, f'{id_exp}, {n_labeled}, {np.round(acc, 2)}, {np.round(adv_acc, 2)}')
    # tf_summary_writer = tf.summary.create_file_writer('tfboard')
    # with tf_summary_writer.as_default():
    #     tf.summary.scalar('accuracy', acc, step=n_labeled)
    print("round 0 time: {:.2f} s".format(time.time() - t))

    rd = 1
    while n_labeled < n_final_labeled:
        print(f"Round {rd}")
    
        # query
        print('>querying...')
        extra_data = None
        if strategy.pseudo_labeling:
            query_idxs, extra_data = strategy.query(n_query)
        else:
            query_idxs = strategy.query(n_query)
        # update
        print('>updating...')
        strategy.update(query_idxs, extra_data)

        print('training...')
        strategy.train()

        # calculate accuracy
        print('evaluation...')
        acc = strategy.eval_acc()
        adv_acc = strategy.eval_adv_acc()
        strategy.eval_dis()

        n_labeled = strategy.dataset.n_labeled()
        print(f"Round {rd}:{n_labeled} testing accuracy: {acc}")
        log_to_file(ACC_FILENAME, f'{id_exp}, {n_labeled}, {np.round(acc, 2)}, {np.round(adv_acc, 2)}')
        # with tf_summary_writer.as_default():
        #     tf.summary.scalar('accuracy', acc, step=n_labeled)
        rd += 1 
    T = time.time() - start
    print(f'Total time: {T/60:.2f} mins.')
    log_to_file('time.txt', f'Total time({ACC_FILENAME}): {T/60:.2f} mins.\n')
