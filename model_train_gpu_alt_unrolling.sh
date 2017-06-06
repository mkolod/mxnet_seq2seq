#!/bin/bash
python seq2seq_bucketing.py --num-hidden 512 --num-embed 512 --num-layers 2 --gpus 0 --batch-size 128 \
  --optimizer adagrad --lr 0.141 --disp-batches 100 --num-epochs 1 --model-prefix trained_model \
  --dropout 0.3 --seed 1234

# --inference-unrolling-for-training

