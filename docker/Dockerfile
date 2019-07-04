FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    sudo \
    git \
    zip \
    libopencv-dev \
    build-essential libssl-dev libbz2-dev libreadline-dev libsqlite3-dev curl \
    wget && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ARG UID
RUN useradd docker -l -u $UID -G sudo -s /bin/bash -m
RUN echo 'Defaults visiblepw' >> /etc/sudoers
RUN echo 'docker ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER docker

ENV PYENV_ROOT /home/docker/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN curl -L https://raw.githubusercontent.com/yyuu/pyenv-installer/master/bin/pyenv-installer | bash

ENV PYTHON_VERSION 3.6.8
RUN pyenv install ${PYTHON_VERSION} && pyenv global ${PYTHON_VERSION}

RUN pip install -U pip setuptools
# for pycocotools
RUN pip install Cython==0.29.1 numpy==1.15.4

COPY requirements/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# mount YOLOv3-in-PyTorch to /work
WORKDIR /work

ENTRYPOINT ["/bin/bash"]
