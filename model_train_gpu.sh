#!/bin/bash
python seq2seq_bucketing.py --num-hidden 512 --num-embed 512 --num-layers 2 --gpus 0,1 --batch-size 256 \
  --optimizer adagrad --lr 0.01 --disp-batches 2 --num-epochs 1 \
  --dropout 0.3 --seed 1234 --model-prefix trained_model --model-prefix trained_model 
# --input-feed --remove-state-feed
#  --model-prefix trained_model
# --use-cudnn-cells 
#  --inference-unrolling-for-training
