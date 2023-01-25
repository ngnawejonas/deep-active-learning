#!/bin/bash

cd src
	
# Run experiment
# Run experiment
python3 main.py \
    --cpus-per-trial 4 \
    --dataset mnist \
    --project-name aal_baseline
# python3 main.py  --cpus-per-trial 4 --project-name aal_base_0
