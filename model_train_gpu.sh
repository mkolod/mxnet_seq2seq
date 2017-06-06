#!/bin/bash
python seq2seq_bucketing.py --num-hidden 512 --num-embed 512 --num-layers 2 --gpus 0,1 --batch-size 256 \
  --optimizer adagrad --lr 0.01 --disp-batches 1 --num-epochs 10 \
  --dropout 0.3 --seed 1234 --model-prefix trained_model --model-prefix trained_model --attention --input-feed --remove-state-feed
# --attention
# --input-feed --remove-state-feed
#  --model-prefix trained_model
# --use-cudnn-cells 
#  --inference-unrolling-for-training
