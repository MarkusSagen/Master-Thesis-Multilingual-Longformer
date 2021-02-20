
# https://hub.docker.com/r/huggingface/transformers-pytorch-gpu/dockerfile
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG PKG_DIR
ARG PRIVATE_DEPS

WORKDIR /workspace

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   wget \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

# RUN apt-get update && apt-get install -y git

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    torch

#RUN git clone https://github.com/NVIDIA/apex
#RUN cd apex && \
#    python3 setup.py install && \
#    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


# Install packages from private repositories
COPY ${PKG_DIR}/ /pkg/
RUN if [ "${PRIVATE_DEPS}" != "none" ]; then \
	for pkg in /pkg/*/* ; \
	do pip install -e $pkg ; \
	done; \
	fi


# Fix permissions
RUN chmod 0777 /workspace
RUN mkdir /.local && chmod 0777 /.local
RUN mkdir /.jupyter && chmod 0777 /.jupyter
RUN mkdir /.cache && chmod 0777 /.cache
# Workaround for transformers library permissions
RUN mkdir /.config && chmod 0777 /.config

# Install python packages
ADD src ./src
ADD requirements.txt .
RUN pip install -r requirements.txt
