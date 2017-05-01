#!/bin/bash
python seq2seq_bucketing.py --num-hidden 512 --num-embed 512 --num-layers 2 --gpus 0 --batch-size 64 \
  --optimizer adagrad --lr 0.1 --disp-batches 1 --num-epochs 1 --model-prefix trained_model --dropout 0.0 
#  --inference-unrolling-for-training
# --use-cudnn-cells 
