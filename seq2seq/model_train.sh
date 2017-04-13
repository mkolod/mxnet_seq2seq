#!/bin/bash
python seq2seq_bucketing.py --num-hidden 512 --num-embed 512 --num-layers 2 --gpus 0 --batch-size 64 --optimizer adam --disp-batches 8 --num-epochs 5 --dropout 0.5
