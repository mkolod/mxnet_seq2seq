#!/bin/bash
python seq2seq_bucketing.py --num-hidden 512 --num-embed 512 --num-layers 2 --gpus 0 --batch-size 32 \
  --optimizer adagrad --lr 0.00141 --disp-batches 10 --num-epochs 10 --model-prefix trained_model \
  --dropout 0.3 --seed 1234 #--remove-state-feed
#  --load-epoch 7
# --use-cudnn-cells 
#  --inference-unrolling-for-training
