FROM nvidia/cuda:8.0-cudnn5-devel

RUN apt-get update && apt-get -y upgrade && \
  apt-get install -y \
  git \
  libopenblas-dev \
  libopencv-dev \
  python-dev \
  python-numpy \
  python-setuptools \
  wget \
  python-pip \
  unzip \ 
  sudo \
  vim

# Build MxNet for Python
RUN cd /root && git clone --recursive https://github.com/dmlc/mxnet.git && cd mxnet && git checkout 6e81d76e6830b70a4a2278ebc08e9d3e3af1c937 && \
# https://github.com/dmlc/mxnet && cd mxnet && \
  cp make/config.mk . && \
  sed -i 's/USE_BLAS = atlas/USE_BLAS = openblas/g' config.mk && \
  sed -i 's/USE_CUDA = 0/USE_CUDA = 1/g' config.mk && \
  sed -i 's/USE_CUDA_PATH = NONE/USE_CUDA_PATH = \/usr\/local\/cuda/g' config.mk && \
  sed -i 's/USE_CUDNN = 0/USE_CUDNN = 1/g' config.mk && \
  make -j"$(nproc)"

# Python3 support
RUN apt-get -y install python3-pip
RUN pip3 install numpy

# Jupyter notebook support
RUN pip install jupyter
COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888

ENV PYTHONPATH /root/mxnet/python

# Build MxNet for Scala
#RUN apt-get -y install maven openjdk-8-jdk scala
#RUN cd /root/mxnet && make scalapkg && make scalainstall

# Build MxNet for R - WIP !!!
#RUN apt-get -y install r-base r-base-dev

WORKDIR /root/mxnet
