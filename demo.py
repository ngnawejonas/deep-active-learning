import argparse
import time
import yaml
from pprint import pprint
import numpy as np
import torch
from utils import get_dataset, get_net, get_strategy, log_to_file


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--cfg_file', type=str, default=None, help="config file")
    parser.add_argument('--seed', type=int, default=1, help="random seed")
    parser.add_argument(
        '--n_init_labeled',
        type=int,
        default=10000,
        help="number of init labeled samples")
    parser.add_argument('--n_query', type=int, default=1000,
                        help="number of queries per round")
    parser.add_argument('--n_round', type=int, default=10, help="number of rounds")
    parser.add_argument(
        '--n_final_labeled',
        type=int,
        default=None,
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
    # parser.add_argument(
    #     '--net_architect',
    #     type=str,
    #     default="None",
    #     choices=[
    #         "None",
    #         "resnet18",
    #         "vgg16"],
    #     help="network architecture")
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
                                 "AdversarialDeepFool"], help="query strategy")
    args = parser.parse_args()

    if args.cfg_file:
        try:
            with open(args.cfg_file, 'r') as config_file:
                config = yaml.load(config_file, Loader=yaml.SafeLoader)
        except yaml.YAMLError as exc:
            print("Error in configuration file:", exc)
        pprint(config)

        seed =  config['seed']
        n_init_labeled = config['n_init_labeled']
        n_query = config['n_query']
        n_final_labeled = config['n_final_labeled']
        if args.n_final_labeled:
            n_round = (n_final_labeled - n_init_labeled) // n_query
        else:
            n_round = config['n_round']

        dataset_name = config['dataset_name']
        strategy_name =  config['strategy_name']

    else:
        pprint(vars(args))
        seed =  args.seed
        n_init_labeled = args.n_init_labeled
        n_query = args.n_query
        n_final_labeled = args.n_final_labeled
        if n_final_labeled:
            n_round = (n_final_labeled - n_init_labeled) // n_query
        else:
            n_round = args.n_round
        n_round = args.n_round
        dataset_name = args.dataset_name
        strategy_name =  args.strategy_name
    print()
    #
    try:
        with open('strategy_config.yaml', 'r') as config_file:
            strategy_config = yaml.load(config_file, Loader=yaml.SafeLoader)
    except yaml.YAMLError as exc:
        print("Error in configuration file:", exc)

    # fix random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.enabled = False

    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = get_dataset(dataset_name)                   # load dataset
    net = get_net(dataset_name, device)                   # load network
    params = {}
    if strategy_config.get(strategy_name):
        params = strategy_config[strategy_name]
    strategy = get_strategy(strategy_name)(dataset, 
                            net, **params)                      # load strategy

    # start experiment
    dataset.initialize_labels(n_init_labeled)
    print(f"number of labeled pool: {n_init_labeled}")
    print(f"number of unlabeled pool: {dataset.n_pool-n_init_labeled}")
    print(f"number of testing pool: {dataset.n_test}")
    print()
    start = time.time()
    # round 0 accuracy
    print("Round 0")
    strategy.train()
    preds = strategy.predict(dataset.get_test_data())
    print(f"Round 0 testing accuracy: {dataset.cal_test_acc(preds)}")

    for rd in range(1, n_round + 1):
        print(f"Round {rd}")

        if strategy.pseudo_labeling:
            # query
            query_idxs, extra_data = strategy.query(n_query)
            # update labels
            strategy.update(query_idxs)    
            strategy.add_extra(extra_data)
        else:
            # query
            query_idxs = strategy.query(n_query)
            # update labels
            strategy.update(query_idxs)

        strategy.train()

        # calculate accuracy
        preds = strategy.predict(dataset.get_test_data())
        print(f"Round {rd} testing accuracy: {dataset.cal_test_acc(preds)}")
    T = time.time() - start
    print(f'Total time: {T} secs.')
    log_to_file('time.txt', f'Total time: {T} secs.')
