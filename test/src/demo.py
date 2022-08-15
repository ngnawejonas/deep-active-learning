import time
import os
import sys
import argparse

import random
import numpy as np
import torch
import torchmetrics
import yaml
from ray import tune
from ray.tune import CLIReporter

from demo_models import CIFAR10_Net, MNIST_Net
from demo_data import get_CIFAR10, get_MNIST
from demo_train import train, test

import wandb
from tqdm import tqdm

# import tensorflow as tf

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


def run_trial(
    config: dict, params: dict, args: argparse.Namespace, num_gpus: int = 0
) -> None:
    """Train a single model according to the configuration provided.

    :param config: The trial and model configuration.
    :param params: The hyperparameters.
    :param args: The program arguments.
    """

    # seed_everything(config["seed"], workers=True)
    set_seeds(config["seed"])
    #
    if args.dry_run:
        wandb.init(project=args.project_name, mode="disabled")
    else:
        wandb.init(project=args.project_name)
    #
    ckpath = 'checkpoints'
    if not os.path.exists(ckpath):
        os.makedirs(ckpath)
    # device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'Using GPU: {use_cuda}')

    train_data, val_data, test_data = get_CIFAR10()
    net = CIFAR10_Net()

    start = time.time()
    train(net, train_data, val_data, params, device)
    print("train time: {:.2f} s".format(time.time() - start))

    test_accuracy = torchmetrics.Accuracy().to(device)
    test_acc = test(net, test_data, test_accuracy, params, device)
    wandb.log({'test acc': test_acc})
    print(f"Test accuracy: {test_acc}")

    T = time.time() - start
    print(f'Total time: {T/60:.2f} mins.')


def run_experiment(params: dict, args: argparse.Namespace) -> None:
    """Run the experiment using Ray Tune.

    :param params: The hyperparameters.
    :param args: The program arguments.
    """
    config = {
        "epochs": tune.grid_search(params["epochs"]),
        "init_labelled_size": tune.grid_search(params["init_labelled_size"]),
        "seed": tune.grid_search(params["seeds"]),
        "lr_schedule": tune.grid_search(params["lr_schedules"]),
    }

    reporter = CLIReporter(
        parameter_columns=["seed"],
        metric_columns=["val_acc"],
    )

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    gpus_per_trial = 1 if use_cuda else 0

    tune.run(
        tune.with_parameters(
            run_trial, params=params, args=args, num_gpus=gpus_per_trial
        ),
        resources_per_trial={"cpu": args.cpus_per_trial, "gpu": gpus_per_trial},
        metric="val_acc",
        mode="max",
        config=config,
        progress_reporter=reporter,
        name=args.project_name,
    )


def main(args: list) -> None:
    """Parse command line args, load training params, and initiate training.

    :param args: command line parameters as list of strings.
    """
    args = parse_args(args)
    params = yaml.safe_load(open("params.yaml"))

    run_experiment(params, args)


def run() -> None:
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`
    This function can be used as entry point to create console scripts.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    run()