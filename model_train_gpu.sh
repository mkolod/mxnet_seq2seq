#!/bin/bash
python seq2seq_bucketing.py --num-hidden 512 --num-embed 512 --num-layers 2 --gpus 1 --batch-size 128 \
  --optimizer adagrad --lr 0.0141 --disp-batches 10 --num-epochs 1 \
  --dropout 0.3 --seed 1234 --model-prefix trained_model --model-prefix trained_model --use-cudnn-cells 
#--attention --remove-state-feed --input-feed
# --attention
# --input-feed --remove-state-feed
#  --model-prefix trained_model
# --use-cudnn-cells 
#  --inference-unrolling-for-training
