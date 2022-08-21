.DEFAULT_GOAL := help

NAME	:= resnet-demo
TAG    	:= $(shell git log -1 --format=%h)
IMG    	:= ${NAME}:${TAG}
LATEST	:= ${NAME}:latest

help:			## Show this help dialog.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)

create-venv:		## Create an isolated virtual environment for this experiment.
	virtualenv env -p python3.9
	. env/bin/activate && pip install -r requirements.txt

build:			## Build the Docker image.
	DOCKER_BUILDKIT=1 docker build --build-arg WANDB_API_KEY=$(WANDB_API_KEY) --build-arg WANDB_TAGS=${IMG} -t ${IMG} .
	docker tag ${IMG} ${LATEST}

run:			## Run the experiment (using gpus=0,1 to specify GPU visibility).
	docker run --user "15455:110" -it --rm --runtime=nvidia --ipc=host \
		-e NVIDIA_VISIBLE_DEVICES=$(gpus) \
		-v /home-local2/jongn2.extra.nobkp:/home-local2/jongn2.extra.nobkp \
		${LATEST} \
		bash src/run.sh

run-lambda:		## Run the experiment on a LambdaStack server (using gpus=0,1 to specify GPU visibility).
	docker run --user "15455:110" -it --rm --ipc=host \
		--gpus '"device=$(gpus)"' \
		-v /home-local2/jongn2.extra.nobkp:/home-local2/jongn2.extra.nobkp \
		${LATEST} \
		bash src/run.sh
