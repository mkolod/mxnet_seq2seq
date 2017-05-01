#!/bin/bash
python seq2seq_bucketing.py --num-hidden 512 --num-embed 512 --num-layers 2 --gpus 0,1 --batch-size 256 --disp-batches 1 --num-epochs 1 --model-prefix trained_model --dropout 0.0 --infer --load-epoch 6 
# --use-cudnn-cells
