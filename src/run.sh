#!/bin/bash

cd src
	
# Run experiment
# Run experiment
python3 main.py \
    --cpus-per-trial 4 \
    --dataset mnist \
    --project-name aal_2024_debug \
    --no-ray
    # --dry-run
# python3 main.py  --cpus-per-trial 4 --dataset mnist --project-name aal_base_AfterDark
