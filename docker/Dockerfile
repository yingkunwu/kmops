ARG PYTORCH="2.5.1"
ARG CUDA="12.4"
ARG CUDNN="9"
FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

# avoid selecting 'Geographic area' during installation
ARG DEBIAN_FRONTEND=noninteractive

# apt install required packages
RUN apt-get update \
    && apt-get install -y ffmpeg libsm6 libxext6 ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    git wget sudo htop \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install required packages from requirements.txt
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN python -m pip install lightning
RUN pip install hydra-core --upgrade
RUN pip install -U 'wandb>=0.12.10'

WORKDIR /workspace