#!/bin/bash
python seq2seq_bucketing.py --num-hidden 512 --num-embed 512 --num-layers 2 --gpus 0 --batch-size 128 \
  --optimizer adagrad --lr 0.0141 --disp-batches 10 --num-epochs 12 \
  --dropout 0.3 --seed 1234 --remove-state-feed
#  --model-prefix trained_model
# --use-cudnn-cells 
#  --inference-unrolling-for-training
