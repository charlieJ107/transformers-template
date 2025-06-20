FROM mcr.microsoft.com/devcontainers/base:noble
# Base image is Ubuntu 24.04

ARG USERNAME=ubuntu

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O cuda-keyring.deb \
    && sudo dpkg -i cuda-keyring.deb \
    && rm cuda-keyring.deb\
    && sudo apt update \
    && sudo apt install -y cuda-toolkit build-essential git curl wget 

USER $USERNAME
WORKDIR /home/$USERNAME
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN echo 'eval "$(uv generate-shell-completion bash)"' >> ~/.bashrc \
    && echo 'eval "$(uv generate-shell-completion zsh)"' >> ~/.zshrc
