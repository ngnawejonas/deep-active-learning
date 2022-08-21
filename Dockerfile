# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

ARG WANDB_API_KEY
ARG WANDB_TAGS

ENV PATH="${PATH}:/home/docker/.local/bin"

# Set timezone
ENV TZ=Canada/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Configure Ray Tune results directory
ENV TUNE_RESULT_DIR="/home-local2/jongn2.extra.nobkp/ray_results"

# Configure Weights & Biases
ENV WANDB_API_KEY=${WANDB_API_KEY}
ENV WANDB_DIR="/home-local2/jongn2.extra.nobkp/wandb"
ENV WANDB_CACHE_DIR="/home-local2/jongn2.extra.nobkp/.cache/wandb"
ENV WANDB_TAGS=${WANDB_TAGS}

RUN apt-get update && \
    apt-get install -y sudo

# Copy files to a new 'work' directory
RUN mkdir -p /home/docker/workspace/
COPY --chown=15455:110 ./ /home/docker/workspace/

# Install Dependencies
RUN sudo apt-get update && \
    sudo apt-get install -y openssh-client git python3.9 python3-pip
RUN mkdir -p -m 0600 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN pip3 install -r /home/docker/workspace/requirements.txt

# Add (and switch to) new 'docker' user
RUN groupadd -g 110 usergroup
RUN adduser --disabled-password --gecos '' --uid 15455 --gid 110 docker
RUN adduser docker sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER docker

# Transfer ownership of home directory
RUN sudo chown -hR docker /home/docker

# Create mount point for large file storage drive
RUN sudo mkdir -p /home-local2/jongn2.extra.nobkp && \
    sudo chown -R docker /home-local2/jongn2.extra.nobkp

# Set the working directory
WORKDIR /home/docker/workspace/
