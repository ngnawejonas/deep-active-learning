# from dis import dis
from pprint import pprint
import os
import argparse
import sys
import time
import random
import yaml
import numpy as np
import torch
from ray import tune
from ray.tune import CLIReporter
import wandb
import seaborn as sns

from main_utils import get_dataset, get_net, get_strategy
from utils import log_to_file, DMAX


def parse_args(args: list) -> argparse.Namespace:
    """Parse command line parameters.

    :param args: command line parameters as list of strings (for example
        ``["--help"]``).
    :return: command line parameters namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train the models for this experiment."
    )

    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="quickly check a single pass",
    )
    parser.add_argument(
        "--no-ray",
        action="store_false",
        default=False,
        help="run without ray",
    )
    parser.add_argument(
        "--dataset-path",
        default="/home-local2/jongn2.extra.nobkp/data",
        help="the path to the dataset",
        type=str,
    )
    parser.add_argument(
        "--cpus-per-trial",
        default=1,
        help="the number of CPU cores to use per trial",
        type=int,
    )
    parser.add_argument(
        "--project-name",
        help="the name of the Weights and Biases project to save the results",
        # required=True,
        type=str,
    )

    return parser.parse_args(args)


def set_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False


def logdist_metrics(dist_list, name, rd, n_labeled):
    logdict = {'avg '+name: np.mean(dist_list),
               'min '+name: np.min(dist_list),
               'max '+name: np.max(dist_list),
               'median '+name: np.median(dist_list),
               'round ': rd,
               'n_labeled': n_labeled}
    return logdict


def logdist_hist(dist_list, name, rd, n_labeled):
    kdefig = sns.displot(data=dist_list.ravel())
    logdict = {name: kdefig.figure}
    return logdict


def dis_report(dis_list, name, rd, n_labeled, correct_idxs=None):
    name = name+'(SUBSET)' if correct_idxs else name+''
    if correct_idxs:
        # wandb.log(logdist_hist(dis_list[correct_idxs], name, rd, n_labeled))
        wandb.log(logdist_metrics(dis_list[correct_idxs], name, rd, n_labeled))
    else:
        # wandb.log(logdist_hist(dis_list, name, rd, n_labeled))
        wandb.log(logdist_metrics(dis_list, name, rd, n_labeled))


def dis_eval_and_report(strategy, rd):
    n_labeled = strategy.dataset.n_labeled()
    dis_list, nb_iter_list, correct_idxs = strategy.eval_test_dis()

    def dis_report_wrap(correct_idxs=None):
        dis_report(dis_list['d_inf'], 'norm inf', rd, n_labeled, correct_idxs)
        dis_report(dis_list['d_2'], 'norm 2', rd, n_labeled, correct_idxs)
        dis_report(nb_iter_list, 'nb iters', rd, n_labeled, correct_idxs)
        dis_report(dis_list['cumul_inf'], 'cumul norm inf',
                   rd, n_labeled, correct_idxs)
        dis_report(dis_list['cumul_2'], 'cumul norm 2',
                   rd, n_labeled, correct_idxs)
    # dis_report_wrap()
    dis_report_wrap(correct_idxs)


def acc_eval_and_report(strategy, rd, logfile, id_exp):
    n_labeled = strategy.dataset.n_labeled()
    test_acc = strategy.eval_acc()
    wandb.log({'clean accuracy (10000)': test_acc,
              'round ': rd, 'n_labeled': n_labeled})
    adv_acc = strategy.eval_adv_acc()
    advkey = 'adversarial accuracy({})'.format(strategy.dataset.n_adv_test)
    wandb.log({advkey: adv_acc, 'round ': rd, 'n_labeled': n_labeled})
    if strategy.dataset.n_adv_test < strategy.dataset.n_test:
        acc2 = strategy.eval_acc_on_adv_test_data()
        acc2key = 'clean accuracy({})'.format(strategy.dataset.n_adv_test)
        wandb.log({acc2key: acc2, 'round ': rd, 'n_labeled': n_labeled})

    print(f"Round {rd}:{n_labeled} testing accuracy: {test_acc}")
    # log_to_file(logfile, f'{id_exp}, {n_labeled}, {np.round( test_acc,  2)}, {np.round(adv_acc, 2)}')
    return test_acc


def eval_and_report(strategy, rd, logfile, id_exp, no_ray=False):
    if not no_ray:
        tune.report(round=rd)
    test_acc = acc_eval_and_report(strategy, rd, logfile, id_exp)
    dis_eval_and_report(strategy, rd)
    return test_acc


def run_trial_empty(
    config: dict, params: dict, args: argparse.Namespace, num_gpus: int = 0
) -> None:
    print("DO NOTHING AND EXIT")


def run_trial(
    config: dict, params: dict, args: argparse.Namespace, num_gpus: int = 0
) -> None:
    """Train a single model according to the configuration provided.

    :param config: The trial and model configuration.
    :param params: The hyperparameters.
    :param args: The program arguments.
    """

    strategy_name_on_file = config['strategy_name'] + \
        "AdvTrain" if params['advtrain_mode'] else config['strategy_name']
    ACC_FILENAME = '{}_{}_{}_{}_{}_{}.txt'.format(
        strategy_name_on_file, params['n_final_labeled'], params['dataset_name'], params['net_arch'], params['n_final_labeled'], 'r'+str(params['repeat']))
    #
    resultsDirName = 'results'
    try:
        os.mkdir(resultsDirName)
        print("Results directory ", resultsDirName,  " Created ")
    except FileExistsError:
        print("Results directory ", resultsDirName,  " already exists")

    # fix random seed
    set_seeds(config['seed'])
    if args.dry_run:
        wandb.init(project=args.project_name, mode="disabled")
    else:
        id = 0 if args.no_ray else tune.get_trial_id()
        exp_name = '{}_run_{}_{}_seed{}'.format(
            config['dataset_name'], id, config['strategy_name'], config['seed'])
        wandb.init(project=args.project_name, name=exp_name, config=config)
    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using GPU: {use_cuda}')
    # print('getting dataset...')
    dataset = get_dataset(params['dataset_name'],
                          params['pool_size'], params['n_adv_test'])
    # print('dataset loaded')
    net = get_net(params, device)

    xparams = dict()
    if params.get(config['strategy_name']):
        xparams = params.get(config['strategy_name'])
        if xparams.get('norm') and xparams.get('norm') == 'np.inf':
            xparams['norm'] = np.inf
        xparams['pseudo_labeling'] = params['pseudo_labelling']
    xparams['dist_file_name'] = 'dist_'+ACC_FILENAME
    id_exp = 0 if args.no_ray else int(tune.get_trial_id().split('_')[-1])
    xparams['id_exp'] = id_exp
    pprint(xparams)

    strategy = get_strategy(config['strategy_name'])(
        dataset, net, **xparams)       # load strategy

    if hasattr(strategy, 'n_subset_ul'):
        strategy.check_querying(params['n_query'])

    # start experiment
    dataset.initialize_labels(params['n_init_labeled'])
    print(f"size of labeled pool: {params['n_init_labeled']}")
    print(f"size of unlabeled pool: {dataset.n_pool-params['n_init_labeled']}")
    print(f"size of testing pool: {dataset.n_test}")
    print()
    start = time.time()
    # round 0 accuracy
    rd = 0
    print("Round 0")
    t = time.time()
    strategy.train()
    print("train time: {:.2f} s".format(time.time() - t))
    print('testing...')
    test_acc = eval_and_report(strategy, rd, ACC_FILENAME, id_exp)
    print("round 0 time: {:.2f} s".format(time.time() - t))

    while strategy.dataset.n_labeled() < params['n_final_labeled']:
        rd = rd + 1
        # query
        print('>querying...')
        extra_data = None
        if strategy.pseudo_labeling:
            query_idxs, extra_data = strategy.query(params['n_query'])
        else:
            query_idxs = strategy.query(params['n_query'])
        # update
        print('>updating...')
        strategy.update(query_idxs, extra_data)

        print('training...')
        strategy.train()

        # calculate accuracy
        print('evaluation...')

        test_acc = eval_and_report(strategy, rd, ACC_FILENAME, id_exp)

    T = time.time() - start
    print(f'Total time: {T/60:.2f} mins.')
    log_to_file('time.txt', f'Total time({ACC_FILENAME}): {T/60:.2f} mins.\n')

    if not args.no_ray:
        tune.report(final_acc=test_acc)


def run_experiment(params: dict, args: argparse.Namespace) -> None:
    """Run the experiment using Ray Tune.

    :param params: The hyperparameters.
    :param args: The program arguments.
    """
    config = {
        "dataset_name": params['dataset_name'],
        "strategy_name": tune.grid_search(params["strategies"]),
        "seed": tune.grid_search(params["seeds"]),
    }
    if args.dry_run:
        config = {
            "strategy_name": 'RandomSampling',
            "seed": 42,
        }
        params['epochs'] = 2
    if args.no_ray:
        run_trial(params=params, args=args, num_gpus=gpus_per_trial)
    else:
        reporter = CLIReporter(
            parameter_columns=["seed", "strategy_name", "dataset_name"],
            metric_columns=["round"],
        )

        use_cuda = not args.no_cuda and torch.cuda.is_available()
        gpus_per_trial = 1 if use_cuda else 0

        tune.run(
            tune.with_parameters(
                run_trial, params=params, args=args, num_gpus=gpus_per_trial
            ),
            resources_per_trial={
                "cpu": args.cpus_per_trial, "gpu": gpus_per_trial},
            # metric="val_acc",
            # mode="max",
            config=config,params, args)
            progress_reporter=reporter,
            name=args.project_name,
        )


def main(args: list) -> None:
    """Parse command line args, load training params, and initiate training.

    :param args: command line parameters as list of strings.
    """
    args = parse_args(args)
    with open('./params2.yaml', 'r') as param_file:
        params = yaml.load(param_file, Loader=yaml.SafeLoader)
    # print(params)
    run_experiment(params, args)


def run() -> None:
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`
    This function can be used as entry point to create console scripts.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
