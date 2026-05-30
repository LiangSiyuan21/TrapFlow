#!/bin/bash
# Step 1: Train RL-based trigger on surrogate model (RF + TAM)
# This trains the LSTM policy network to find optimal trigger insertion positions.
# Output: trigger checkpoint saved under ./checkpoints/

DATA_PATH="./data/sirinam/"
GPU=0

python src/run_train_backdoor_trigger.py \
    --data-path ${DATA_PATH} \
    --verbose --one-fold \
    --model rf --feature-type tam --seq-length 1800 \
    --mode train \
    --lr0 0.000001 --epochs 30 --batch-size 1024 \
    --gpu ${GPU} \
    --asr --return-backdoored \
    --backdoor-ratio 0.5 \
    --backdoor-lable 0 \
    --backdoor-label-type lc \
    --distance levenshtein \
    --trigger-train True \
    --backdoor-length 20000 \
    --backdoor-nums 7 \
    --eval-nums -1
