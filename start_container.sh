#!/bin/bash

nvidia-docker run --rm -it -v `pwd`:/mxnet_seq2seq -p 8888:8888 mxnet_seq2seq
