#!/bin/bash
python seq2seq_bucketing.py --num-hidden 512 --num-embed 512 --num-layers 2 --gpus 0 --batch-size 64 \
  --optimizer adagrad --disp-batches 10 --num-epochs 1 --model-prefix trained_model \
  --dropout 0.3 --seed 1234 --use-cudnn-cells --lr 0.141 --scaling 128.0 # 128.0 # 128.0 # try scaling by 128.0 for HMMA experiments
#  --inference-unrolling-for-training
