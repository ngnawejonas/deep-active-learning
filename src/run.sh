#!/bin/bash

cd src
	
# Run experiment
# Run experiment
python3 main.py \
    --cpus-per-trial 4 \
    --project-name aal_resnet18_cifar10_train
