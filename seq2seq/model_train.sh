#!/bin/bash
python seq2seq_bucketing.py --num-hidden 128 --num-embed 128 --num-layers 2 --gpus 0,1 --batch-size 32 --optimizer adam --disp-batches 8 --num-epochs 5 --dropout 0.5
