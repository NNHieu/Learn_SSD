# FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

# # # Install some basic utilities
# RUN apt-get update && apt-get install -y \
#     curl \
#     ca-certificates \
#     sudo \
#     git \
#     bzip2 \
#     libx11-6 &&\
#     apt-get clean &&\ 
#     rm -rf /var/lib/apt/lists/*


# ARG USER_ID=1001
# ARG GROUP_ID=1001

# RUN addgroup --gid $GROUP_ID user
# RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user && \
#     chown -R user:user /workspace && \
#     echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
# RUN chmod -R 777 /opt/conda
# USER user 

# # # All users can use /home/user as their home directory
# ENV PATH="${PATH}:/home/user/.local/bin"
# ENV HOME=/home/user
# RUN chmod 777 /home/user && chmod 777 /workspace
# RUN conda init

# # Install OpenJDK-11
# RUN sudo apt-get update && \
#     sudo apt-get install -y openjdk-11-jre-headless && \
#     # apt-get install -y ant && \
#     sudo apt-get clean;

# # Fix certificate issues
# RUN sudo apt-get update && \
#     sudo apt-get install ca-certificates-java && \
#     sudo apt-get clean && \
#     sudo update-ca-certificates -f;

# # Setup JAVA_HOME -- useful for docker commandline
# ENV JAVA_HOME /usr/lib/jvm/java-11-openjdk-amd64/
# RUN export JAVA_HOME


# COPY requirements.yml /workspace
# RUN pip install --no-cached-dir -r requirements.yml
# RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager && \
#     jupyter labextension enable widgetsnbextension && \
#     jupyter lab clean
# RUN conda clean -ya
# # RUN conda env update --file env.yml  --prune
# # RUN conda config --add channels conda-forge && \
# #     conda config --set channel_priority strict && \
# #     conda install --file requirements.txt
# # RUN pip install --no-cache-dir -r requirements.txt

# -*- coding: utf-8 -*-
#
# This software may be modified and distributed under the terms
# of the MIT license.  See the LICENSE file for details.
#
# Multistage Dockerfile to build a Python 3.8 image based on Debian Buster.
# Inspired by https://github.com/docker-library/python/blob/master/3.8/buster/slim/Dockerfile
# Notable changes:
# - adapted to multistage build
# - removed cleanup steps in builder image for readibility
# - build without tk, ncurses and readline

ARG BASE_IMAGE_NAME=nvidia/cuda:11.0-base

# Intermediate build container
FROM $BASE_IMAGE_NAME AS builder

ENV PYTHON_VERSION 3.8.6
ENV PYTHON_PIP_VERSION 20.2.4

ARG BASE_IMAGE_NAME
RUN echo "Using base image \"${BASE_IMAGE_NAME}\" to build Python ${PYTHON_VERSION}"

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

ENV GPG_KEY E3FF2839C048B25C084DEBE9B26995E310250568
# https://github.com/pypa/get-pip
ENV PYTHON_GET_PIP_URL https://bootstrap.pypa.io/get-pip.py
ENV PYTHON_GET_PIP_SHA256 b3153ec0cf7b7bbf9556932aa37e4981c35dc2a2c501d70d91d2795aa532be79

RUN set -ex && \
	apt-get update && apt-get install --assume-yes --no-install-recommends \
		ca-certificates \
		dirmngr \
		dpkg-dev \
		gcc \
		gnupg \
		libbz2-dev \
		libc6-dev \
		libexpat1-dev \
		libffi-dev \
		liblzma-dev \
		libsqlite3-dev \
		libssl-dev \
		make \
		netbase \
		uuid-dev \
		wget \
		xz-utils \
		zlib1g-dev

RUN wget --no-verbose --output-document=python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz" \
	&& wget --no-verbose --output-document=python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc" \
	&& export GNUPGHOME="$(mktemp -d)" \
	&& gpg --batch --keyserver ha.pool.sks-keyservers.net --recv-keys "$GPG_KEY" \
	&& gpg --batch --verify python.tar.xz.asc python.tar.xz \
	&& { command -v gpgconf > /dev/null && gpgconf --kill all || :; } \
	&& rm -rf "$GNUPGHOME" python.tar.xz.asc \
	&& mkdir -p /usr/src/python \
	&& tar -xJC /usr/src/python --strip-components=1 -f python.tar.xz \
	&& rm python.tar.xz

RUN cd /usr/src/python \
	&& gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)" \
	&& ./configure --help \
	&& ./configure \
		--build="$gnuArch" \
		--prefix="/python" \
		--enable-loadable-sqlite-extensions \
		--enable-optimizations \
		--enable-ipv6 \
		--disable-shared \
		--with-system-expat \
		--without-ensurepip \
	&& make -j "$(nproc)" \
	&& make install

RUN strip /python/bin/python3.8 && \
	strip --strip-unneeded /python/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.a && \
	strip --strip-unneeded /python/lib/python3.8/lib-dynload/*.so && \
	rm /python/lib/libpython3.8.a && \
	ln /python/lib/python3.8/config-3.8-x86_64-linux-gnu/libpython3.8.a /python/lib/libpython3.8.a

# install pip
RUN set -ex; \
	\
	wget --no-verbose --output-document=get-pip.py "$PYTHON_GET_PIP_URL"; \
	# echo "$PYTHON_GET_PIP_SHA256 *get-pip.py" | sha256sum --check --strict -; \
	\
	/python/bin/python3 get-pip.py \
		--disable-pip-version-check \
		--no-cache-dir \
		"pip==$PYTHON_PIP_VERSION" "wheel"

# cleanup
RUN find /python/lib -type d -a \( \
		-name test -o \
		-name tests -o \
		-name idlelib -o \
		-name turtledemo -o \
		-name pydoc_data -o \
		-name tkinter \) -exec rm -rf {} +

######################################################################
#### Base Image
######################################################################
FROM $BASE_IMAGE_NAME

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*


ENV LANG=C.UTF-8 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=true

###################################
#### Setup Python3
###################################
# make some useful symlinks that are expected to exist
RUN ln -s /python/bin/python3-config /usr/local/bin/python-config && \
	ln -s /python/bin/python3 /usr/local/bin/python && \
	ln -s /python/bin/python3 /usr/local/bin/python3 && \
	ln -s /python/bin/pip3 /usr/local/bin/pip && \
	ln -s /python/bin/pip3 /usr/local/bin/pip3 && \
	# install depedencies
	apt-get update && \
	apt-get install --assume-yes --no-install-recommends ca-certificates libexpat1 libsqlite3-0 libssl1.1 && \
	apt-get purge --assume-yes --auto-remove -o APT::AutoRemove::RecommendsImportant=false && \
	rm -rf /var/lib/apt/lists/*

# copy in Python environment
COPY --from=builder /python /python
ENV PATH $PATH:/python/bin


###################################
#### Install pytorch
###################################
# CUDA 11.0-specific steps
RUN pip install --no-cache-dir torch==1.7.0+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html

# OpenCV libs
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

###################################
#### Add User
###################################
ARG USER_ID=1000
ARG GROUP_ID=28

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user &&\
    echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user 
ENV PATH $PATH:/home/user/.local/bin
###################################
#### Install Requirements
###################################

# Create a working directory
ARG HOME=/home/user
RUN mkdir ${HOME}/code
WORKDIR ${HOME}/code

COPY requirements.txt ${HOME}/code/
RUN pip install --no-cache-dir -r requirements.txt --user
RUN jupyter nbextension enable --py widgetsnbextension
