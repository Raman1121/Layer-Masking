#!/bin/bash

for i in {1..15}
do
   torchrun --nproc_per_node=1 train.py --model vit_base --epochs 2 --batch-size 64 --opt adamw --lr 0.003 --lr-scheduler cosineannealinglr --lr-warmup-method linear --lr-warmup-epochs 15 --lr-warmup-decay 0.033 --amp --tuning_method tune_attention_random
   echo "FINISHED ITERATION $i"
   echo
done
