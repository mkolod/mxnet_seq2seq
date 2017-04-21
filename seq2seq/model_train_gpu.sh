#!/bin/bash
python seq2seq_bucketing.py --num-hidden 512 --num-embed 512 --num-layers 2 --optimizer adam --disp-batches 8 --batch-size 128 --gpus 0,1 --num-epochs 10 --max-grad-norm 5.0
