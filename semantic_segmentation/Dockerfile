FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt -qq update && apt -qq install -y --no-install-recommends git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/cache/*
RUN apt install -y locales wget

# Unicode support:
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Create user home directory
ENV USER_NAME aicrowd
ENV HOME_DIR /home/$USER_NAME

# Replace HOST_UID/HOST_GUID with your user / group id
ENV HOST_UID 1001
ENV HOST_GID 1001

# Use bash as default shell, rather than sh
ENV SHELL /bin/bash

# Set up user
RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${HOST_UID} \
    ${USER_NAME}

USER ${USER_NAME}
WORKDIR ${HOME_DIR}

ENV CONDA_DIR ${HOME_DIR}/.conda

ENV PATH ${CONDA_DIR}/bin:${PATH}

COPY --chown=1001:1001 requirements.txt ${HOME_DIR}/requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY --chown=1001:1001 . ${HOME_DIR}
