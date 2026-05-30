#!/bin/bash
# Step 4: Evaluate backdoor attack in open-world setting
# Adds unmonitored instances to simulate a realistic scenario.
# Metrics: Clean accuracy + ASR + open-world PR metrics

DATA_PATH="./data/sirinam/"
GPU=0
TRIGGER_PTH="./checkpoints/<your_trigger>/positions_counts_epoch1.json"  # Update this path
BACKDOOR_LENGTH=20000
BACKDOOR_NUM=7
BACKDOOR_LABEL=33
BACKDOOR_RATIO=1.0
UNMON_INST=40000

# ===================== RF (Open-World) =====================
python src/run_attack_backdoor.py \
    --data-path ${DATA_PATH} \
    --verbose --one-fold \
    --mon-classes 95 --page-per-class 1 \
    --mon-inst-train -1 --mon-inst 1000 \
    --unmon-inst ${UNMON_INST} --open-world \
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

# ===================== DF (Open-World) =====================
python src/run_attack_backdoor.py \
    --data-path ${DATA_PATH} \
    --verbose --one-fold \
    --mon-classes 95 --page-per-class 1 \
    --mon-inst-train -1 --mon-inst 1000 \
    --unmon-inst ${UNMON_INST} --open-world \
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

# ===================== VarCNN (Open-World) =====================
python src/run_attack_backdoor.py \
    --data-path ${DATA_PATH} \
    --verbose --one-fold \
    --mon-classes 95 --page-per-class 1 \
    --mon-inst-train -1 --mon-inst 1000 \
    --unmon-inst ${UNMON_INST} --open-world \
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

# ===================== ARES (Open-World) =====================
python src/run_attack_backdoor.py \
    --data-path ${DATA_PATH} \
    --verbose --one-fold \
    --mon-classes 95 --page-per-class 1 \
    --mon-inst-train -1 --mon-inst 1000 \
    --unmon-inst ${UNMON_INST} --open-world \
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
