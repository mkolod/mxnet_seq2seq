#!/bin/bash
python seq2seq_bucketing.py --num-hidden 512 --num-embed 512 --num-layers 1 --gpus 0,1 --batch-size 64 --optimizer adagrad --disp-batches 8 --num-epochs 1 --dropout 0.5
