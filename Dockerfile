FROM nvidia/cuda:8.0-cudnn5-devel

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        wget \
        git \
        cmake \
        libatlas-base-dev \
        libglib2.0-dev \
        libtiff5-dev \
        libjpeg8-dev \
        zlib1g-dev \
        python-dev && \
    rm -rf /var/lib/apt/lists/*

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

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install --upgrade --no-cache-dir numpy setuptools requests

RUN OPENCV_VERSION=3.1.0 && \
    wget -q -O - https://github.com/Itseez/opencv/archive/${OPENCV_VERSION}.tar.gz | tar -xzf - && \
    cd /opencv-${OPENCV_VERSION} && \
    cmake -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr \
          -DWITH_CUDA=OFF -DWITH_1394=OFF \
          -DBUILD_opencv_cudalegacy=OFF -DBUILD_opencv_stitching=OFF -DWITH_IPP=OFF . && \
    make -j"$(nproc)" install && \
    cp lib/cv2.so /usr/local/lib/python2.7/site-packages/ && \
    rm -rf /opencv-${OPENCV_VERSION}

# Build MxNet for Python
RUN cd /root && git clone --recursive https://github.com/dmlc/mxnet.git && cd mxnet && git checkout 6e81d76e6830b70a4a2278ebc08e9d3e3af1c937 && \
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

WORKDIR /root/mxnet

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib
RUN make -j$(nproc) && \
    mv lib/libmxnet.so /usr/local/lib && \
    ldconfig && \
    make clean && \
    cd python && \
    pip install -e .

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
