#!/bin/bash
# Step 2: Generate trigger patterns from the pre-trained policy network
# This loads the pre-trained trigger model and generates insertion positions/counts.
# Output: positions_counts JSON file under ./checkpoints/
#
# We provide a pre-trained RL policy weight (54KB) at:
#   pretrained/trigger_policy.pth
# Use --pretrained to directly specify the weight path.

DATA_PATH="./data/sirinam/"
GPU=0
PRETRAINED="./pretrained/trigger_policy.pth"

# Generate triggers with different lengths
for BACKDOOR_LENGTH in 1000 2000 4000 8000; do
    echo "Generating trigger with length=${BACKDOOR_LENGTH}..."
    python src/run_test_backdoor_trigger.py \
        --data-path ${DATA_PATH} \
        --verbose --one-fold \
        --mon-classes 95 --page-per-class 1 \
        --mon-inst-train -1 --mon-inst 1000 \
        --model rf --feature-type tam --seq-length 1800 \
        --mode train \
        --lr0 0.000001 --epochs 1 --batch-size 512 \
        --gpu ${GPU} \
        --pretrained ${PRETRAINED} \
        --trigger-pretrain True \
        --asr --return-backdoored \
        --backdoor-ratio 0.5 \
        --backdoor-lable 0 \
        --backdoor-label-type lc \
        --distance levenshtein \
        --trigger-train True \
        --backdoor-length ${BACKDOOR_LENGTH} \
        --backdoor-nums 7 \
        --eval-nums -1
done
