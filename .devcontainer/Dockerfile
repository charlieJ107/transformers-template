FROM mcr.microsoft.com/devcontainers/base:noble
# Base image is Ubuntu 24.04

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O cuda-keyring.deb \
    && sudo dpkg -i cuda-keyring.deb \
    && rm cuda-keyring.deb\
    && sudo apt update \
    && sudo apt install -y cuda-toolkit build-essential git curl wget \
    && sudo apt clean && sudo rm -rf /var/lib/apt/lists/* 
 