#!/bin/bash
python seq2seq_bucketing.py --num-hidden 512 --num-embed 512 --num-layers 2 --gpus 0,1 --batch-size 256 --optimizer adagrad --lr 0.1 --disp-batches 10 --num-epochs 10 --model-prefix trained_model --dropout 0.0
# --use-cudnn-cells 
