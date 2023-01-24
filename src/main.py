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
from utils import log_to_file

from query_strategies import AdversarialDeepFool, RandomSampling


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
        action="store_true",
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
    parser.add_argument(
        "--dataset",
        help="dataset used",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--debug-strategy",
        help="the strategy to use in debug mode",
        default="Random",
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

def tune_report(no_ray, **args):
    if not no_ray and tune.is_session_enabled():
        tune.report(**args)

def logdist_metrics(dis_list, name, rd, n_labeled):

    logdict = {'avg '+name: np.mean(dis_list),
               'min '+name: np.min(dis_list),
               'max '+name: np.max(dis_list),
               'median '+name: np.median(dis_list),
               'round ': rd,
               'n_labeled': n_labeled}
    return logdict


def logdist_hist(dis_list, name, rd, n_labeled):
    kdefig = sns.displot(data=dis_list.ravel())
    logdict = {name: kdefig.figure}
    return logdict


def dis_report(dis_list, name, rd, n_labeled, correct_idxs=None):
    dis_list = np.array(dis_list)
    if correct_idxs:
        # wandb.log(logdist_hist(dis_list[correct_idxs], name, rd, n_labeled))
        wandb.log(logdist_metrics(dis_list[correct_idxs], name+'(Cor. Subset)', rd, n_labeled))
        mask = np.ones(len(dis_list), dtype=bool)
        mask[correct_idxs] = False 
        wandb.log(logdist_metrics(dis_list[mask], name+'(InCor. Subset)', rd, n_labeled))
    else:
        # wandb.log(logdist_hist(dis_list, name, rd, n_labeled))
        wandb.log(logdist_metrics(dis_list, name, rd, n_labeled))


def dis_eval_and_report(strategy, rd):
    n_labeled = strategy.dataset.n_labeled()
    print("___dis_eval_and_report___")
    dis_list, nb_iter_list, correct_idxs = strategy.eval_test_dis()

    def dis_report_wrap(correct_idxs=None):
        dis_report(dis_list['d_inf'], 'norm inf', rd, n_labeled, correct_idxs)
        dis_report(dis_list['d_2'], 'norm 2', rd, n_labeled, correct_idxs)
        dis_report(nb_iter_list, 'nb iters', rd, n_labeled, correct_idxs)
        dis_report(dis_list['cumul_inf'], 'cumul norm inf',
                   rd, n_labeled, correct_idxs)
        dis_report(dis_list['cumul_2'], 'cumul norm 2',
                   rd, n_labeled, correct_idxs)
    dis_report_wrap()
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

    print(f"Round {rd}: {n_labeled} clean accuracy: {test_acc} adv accuracy: {adv_acc}")
    # log_to_file(logfile, f'{id_exp}, {n_labeled}, {np.round( test_acc,  2)}, {np.round(adv_acc, 2)}')
    return test_acc


def eval_and_report(strategy, rd, logfile, id_exp, no_ray=False):
    tune_report(no_ray, round=rd)
    test_acc = acc_eval_and_report(strategy, rd, logfile, id_exp)
    # if isinstance(strategy, RandomSampling) or isinstance(strategy, AdversarialDeepFool):
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

    strategy_params = dict()
    if params.get(config['strategy_name']):
        strategy_params = params.get(config['strategy_name'])
        if strategy_params.get('norm'):
            strategy_params['norm'] = float(strategy_params['norm'])
    else:
        strategy_params['pseudo_labeling'] = params['pseudo_labelling']
    strategy_params['max_iter'] = params['max_iter']
    assert strategy_params['max_iter'] is not None
    strategy_params['dist_file_name'] = 'dist_'+ACC_FILENAME
    id_exp = 0 if args.no_ray else int(tune.get_trial_id().split('_')[-1])
    strategy_params['id_exp'] = id_exp
    pprint(strategy_params)
    # if params['test_attack'].get('args'):
    #     params['test_attack']['args']['nb_iter'] = params['max_iter']
    strategy = get_strategy(config['strategy_name'])(dataset, net, **strategy_params)       # load strategy

    if hasattr(strategy, 'n_subset_ul'):
        strategy.check_querying(params['n_query'])

    # start experiment
    dataset.initialize_labels(params['n_init_labeled'])
    print(f"size of labeled pool: {params['n_init_labeled']}")
    print(f"size of unlabeled pool: {dataset.n_pool-params['n_init_labeled']}")
    print(f"size of testing pool: {dataset.n_test}")
    print()
    start = time.time()
    # round 0
    # rd = 0
    # print("Round 0")
    # t = time.time()
    # strategy.train()
    # print("train time: {:.2f} s".format(time.time() - t))
    # print('testing...')
    # test_acc = eval_and_report(strategy, rd, ACC_FILENAME, id_exp)
    # print("round 0 time: {:.2f} s".format(time.time() - t))
    def active_learning_round(rd):
        print(f"Round {rd}")

        if rd != 0:
            # query
            print('>querying...')
            extra_data = None

            if strategy.pseudo_labeling:
                # breakpoint()
                query_idxs, extra_data = strategy.query(params['n_query'])
            else:
                query_idxs = strategy.query(params['n_query'])
            # update
            print('>updating...')
            strategy.update(query_idxs, extra_data)

        print('>training...')
        strategy.train()

        # calculate accuracy
        print('>evaluation...')

        test_acc = eval_and_report(strategy, rd, ACC_FILENAME, id_exp)

        return test_acc

    test_acc = active_learning_round(0)
    #from round 1
    rd = 1
    while strategy.dataset.n_labeled() < params['n_final_labeled']:
        test_acc = active_learning_round(rd)
        rd  = rd + 1

    T = time.time() - start
    print(f'Total time: {T/60:.2f} mins.')
    log_to_file('time.txt', f'Total time({ACC_FILENAME}): {T/60:.2f} mins.\n')

    tune_report(args.no_ray, final_acc=test_acc)


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
            "strategy_name": args.debug_strategy,
            "seed": 42,
        }
        params['epochs'] = 5
        params['max_iter'] = 2

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpus_per_trial = 1 if use_cuda else 0

    if args.no_ray:
        run_trial(config=config, params=params, args=args, num_gpus=gpus_per_trial)
    else:
        reporter = CLIReporter(
            parameter_columns=["seed", "strategy_name", "dataset_name"],
            metric_columns=["round"],
        )

        tune.run(
            tune.with_parameters(
                run_trial, params=params, args=args, num_gpus=gpus_per_trial
            ),
            resources_per_trial={
                "cpu": args.cpus_per_trial, "gpu": gpus_per_trial},
            # metric="val_acc",
            # mode="max",
            config=config,
            progress_reporter=reporter,
            name=args.project_name,
        )


def main(args: list) -> None:
    """Parse command line args, load training params, and initiate training.

    :param args: command line parameters as list of strings.
    """
    args = parse_args(args)
    paramsfilename = f'./params_{args.dataset}.yaml'
    with open(paramsfilename, 'r') as param_file:
        params = yaml.load(param_file, Loader=yaml.SafeLoader)
    run_experiment(params, args)


def run() -> None:
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`
    This function can be used as entry point to create console scripts.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()
