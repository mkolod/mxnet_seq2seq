FROM nvidia/cuda:8.0-cudnn5-devel

RUN apt-get update && apt-get -y upgrade && \
  apt-get install -y \
  build-essential \
  ca-certificates \
  git \
  libopenblas-dev \
  libatlas-base-dev \
  libglib2.0-dev \
  libopencv-dev \
  python-dev \
  python-numpy \
  python-setuptools \
  wget \
  cmake \
  curl \
  python-pip \
  python-dev \
  unzip \ 
  sudo \
  vim \
  libglib2.0-dev \
  libtiff5-dev \
  libjpeg8-dev \
  zlib1g-dev 

RUN pip install --upgrade numpy scipy matplotlib scikit-learn sympy nltk setuptools requests

COPY nan.patch /root

# Build MxNet for Python
RUN cd /root && git clone --recursive https://github.com/dmlc/mxnet.git && cp nan.patch /root/mxnet/ && \
  cd mxnet && git checkout 955f6be6977ca1a27d3e912fd62a08f019dd1f76 && git apply nan.patch && \
  cp make/config.mk . && \
    echo "USE_CUDA=1" >> config.mk && \
    echo "USE_CUDNN=1" >> config.mk && \
    echo "CUDA_ARCH :=" \
         "-gencode arch=compute_35,code=sm_35" \
         "-gencode arch=compute_52,code=sm_52" \
         "-gencode arch=compute_60,code=sm_60" \
         "-gencode arch=compute_61,code=sm_61" \
         "-gencode arch=compute_61,code=compute_61" >> config.mk && \
    echo "USE_CUDA_PATH=/usr/local/cuda" >> config.mk 

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib
RUN cd /root/mxnet && make -j$(nproc) && \
    mv lib/libmxnet.so /usr/local/lib && \
    ldconfig && \
    make clean && \
    cd python && \
    pip install -e .

# Python3 support
RUN apt-get -y install python3-pip
RUN pip3 install numpy

# Jupyter notebook support
COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888

ENV PYTHONPATH /root/mxnet/python

# Build MxNet for Scala
#RUN apt-get -y install maven openjdk-8-jdk scala
#RUN cd /root/mxnet && make scalapkg && make scalainstall

# Build MxNet for R - WIP !!!
#RUN apt-get -y install r-base r-base-dev

RUN pip install unidecode dill tqdm

WORKDIR /root/mxnet



