#!/bin/bash
# Step 3: Evaluate backdoor attack in closed-world setting
# Tests the backdoor attack on different victim models using the generated trigger.
# Metrics: Clean accuracy + Attack Success Rate (ASR)

DATA_PATH="./data/sirinam/"
GPU=0
TRIGGER_PTH="./checkpoints/<your_trigger>/positions_counts_epoch1.json"  # Update this path
BACKDOOR_LENGTH=20000
BACKDOOR_NUM=7
BACKDOOR_LABEL=33
BACKDOOR_RATIO=1.0

# ===================== RF =====================
python src/run_attack_backdoor.py \
    --data-path ${DATA_PATH} \
    --verbose --one-fold \
    --mon-classes 95 --page-per-class 1 \
    --mon-inst-train -1 --mon-inst 1000 \
    --model rf --feature-type tam --seq-length 1800 \
    --epochs 50 --lr0 0.0005 --weight-decay 0.001 --batch-size 200 \
    --gpu ${GPU} \
    --asr --return-backdoored \
    --backdoor-ratio ${BACKDOOR_RATIO} \
    --backdoor-type BackdoorRLNet_optimize_multi_patch_in \
    --trigger-pth ${TRIGGER_PTH} \
    --backdoor-length ${BACKDOOR_LENGTH} --backdoor-num ${BACKDOOR_NUM} \
    --backdoor-label-type lc --backdoor-lable ${BACKDOOR_LABEL} \
    --eval-nums -1

# ===================== DF =====================
python src/run_attack_backdoor.py \
    --data-path ${DATA_PATH} \
    --verbose --one-fold \
    --mon-classes 95 --page-per-class 1 \
    --mon-inst-train -1 --mon-inst 1000 \
    --model df --feature-type df --seq-length 10000 \
    --epochs 50 --lr0 0.002 --weight-decay 0 --batch-size 64 \
    --gpu ${GPU} \
    --asr --return-backdoored \
    --backdoor-ratio ${BACKDOOR_RATIO} \
    --backdoor-type BackdoorRLNet_optimize_multi_patch_in \
    --trigger-pth ${TRIGGER_PTH} \
    --backdoor-length ${BACKDOOR_LENGTH} --backdoor-num ${BACKDOOR_NUM} \
    --backdoor-label-type lc --backdoor-lable ${BACKDOOR_LABEL} \
    --eval-nums -1

# ===================== VarCNN =====================
python src/run_attack_backdoor.py \
    --data-path ${DATA_PATH} \
    --verbose --one-fold \
    --mon-classes 95 --page-per-class 1 \
    --mon-inst-train -1 --mon-inst 1000 \
    --model varcnn --feature-type tiktok --seq-length 10000 \
    --epochs 50 --lr0 0.002 --weight-decay 0.0005 --batch-size 64 \
    --gpu ${GPU} \
    --asr --return-backdoored \
    --backdoor-ratio ${BACKDOOR_RATIO} \
    --backdoor-type BackdoorRLNet_optimize_multi_patch_in \
    --trigger-pth ${TRIGGER_PTH} \
    --backdoor-length ${BACKDOOR_LENGTH} --backdoor-num ${BACKDOOR_NUM} \
    --backdoor-label-type lc --backdoor-lable ${BACKDOOR_LABEL} \
    --eval-nums -1

# ===================== ARES =====================
python src/run_attack_backdoor.py \
    --data-path ${DATA_PATH} \
    --verbose --one-fold \
    --mon-classes 95 --page-per-class 1 \
    --mon-inst-train -1 --mon-inst 1000 \
    --model ares --feature-type df --seq-length 10000 \
    --epochs 50 --lr0 0.002 --weight-decay 0.001 --batch-size 256 \
    --gpu ${GPU} \
    --asr --return-backdoored \
    --backdoor-ratio ${BACKDOOR_RATIO} \
    --backdoor-type BackdoorRLNet_optimize_multi_patch_in \
    --trigger-pth ${TRIGGER_PTH} \
    --backdoor-length ${BACKDOOR_LENGTH} --backdoor-num ${BACKDOOR_NUM} \
    --backdoor-label-type lc --backdoor-lable ${BACKDOOR_LABEL} \
    --eval-nums -1

# ===================== TMWF =====================
python src/run_attack_backdoor.py \
    --data-path ${DATA_PATH} \
    --verbose --one-fold \
    --mon-classes 95 --page-per-class 1 \
    --mon-inst-train -1 --mon-inst 1000 \
    --model tmwf --feature-type df --seq-length 30720 \
    --epochs 50 --lr0 0.002 --weight-decay 0.0005 --batch-size 256 \
    --gpu ${GPU} \
    --asr --return-backdoored \
    --backdoor-ratio ${BACKDOOR_RATIO} \
    --backdoor-type BackdoorRLNet_optimize_multi_patch_in \
    --trigger-pth ${TRIGGER_PTH} \
    --backdoor-length ${BACKDOOR_LENGTH} --backdoor-num ${BACKDOOR_NUM} \
    --backdoor-label-type lc --backdoor-lable ${BACKDOOR_LABEL} \
    --eval-nums -1
