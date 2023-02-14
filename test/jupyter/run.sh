#!/bin/bash

# Launch jupyter notebook
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --no-browser \
  --port=8889 \
  --NotebookApp.port_retries=0