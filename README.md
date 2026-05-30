# WFTransfer: Backdoor Attacks on Website Fingerprinting via Transferable Triggers

This repository contains the implementation of our paper on backdoor attacks against website fingerprinting (WF) classifiers. We propose a reinforcement learning-based approach to generate transferable trigger patterns that can be inserted into encrypted network traffic traces to mislead WF classifiers.

## Overview

Website fingerprinting attacks allow adversaries to identify which websites a user visits through encrypted traffic analysis. Our work explores the vulnerability of WF classifiers to backdoor attacks, where an attacker poisons the training data with trigger-embedded traces to cause targeted misclassification at test time.

**Key components:**
- **RL-based trigger optimization**: An LSTM policy network learns optimal trigger insertion positions and patterns using distance-based rewards (Levenshtein distance).
- **Multi-patch insertion strategy**: Triggers are split into multiple patches and inserted at strategic positions in the traffic trace.
- **Transferable triggers**: Triggers trained on a surrogate model (e.g., RF) transfer effectively to other victim models (DF, VarCNN, ARES, TMWF, etc.).

## Project Structure

```
WFTransfer/
├── README.md
├── requirements.txt
├── LICENSE
├── scripts/                          # Example training/testing scripts
│   ├── train_trigger.sh              # Step 1: Train trigger with RL
│   ├── generate_trigger.sh           # Step 2: Generate trigger patterns
│   ├── attack_closed_world.sh        # Step 3: Backdoor attack (closed-world)
│   └── attack_open_world.sh          # Step 4: Backdoor attack (open-world)
├── pretrained/
│   └── trigger_policy.pth            # Pre-trained RL trigger policy (54KB)
├── src/
│   ├── run_train_backdoor_trigger.py     # Entry: RL trigger training
│   ├── run_test_backdoor_trigger.py      # Entry: Trigger generation
│   ├── run_attack_backdoor.py            # Entry: Backdoor attack evaluation
│   ├── run_attack_backdoor_kf.py         # Entry: K-Fingerprinting attack
│   ├── attacks/
│   │   ├── attack_backdoor.py            # Main backdoor attack pipeline
│   │   ├── attack_backdoor_kf.py         # K-Fingerprinting variant
│   │   ├── train_backdoor_trigger.py     # RL-based trigger training logic
│   │   ├── generate_backdoor_trigger.py  # Trigger generation logic
│   │   ├── attack_model.py               # LSTM policy network
│   │   ├── function.py                   # Helper functions (insertion, sampling, rewards)
│   │   └── modules/                      # WF classifier architectures
│   │       ├── rf.py                     # RF (Recurrent Fingerprinting)
│   │       ├── df.py                     # DF (Deep Fingerprinting)
│   │       ├── varcnn.py                 # Var-CNN
│   │       ├── ares.py                   # ARES
│   │       ├── tmwf.py                   # TMWF
│   │       ├── inception.py              # Inception-based model
│   │       ├── kfingerprinting.py        # K-Fingerprinting
│   │       └── ...
│   └── utils/
│       ├── backdoor_attack_strategy.py   # Backdoor insertion strategies
│       ├── compute_WF_distance.py        # Distance metrics (OSAD, Damerau-Levenshtein, FAST)
│       ├── data.py                       # Dataset loading and preprocessing
│       ├── general.py                    # General utilities
│       ├── metric.py                     # Evaluation metrics (accuracy, ASR, PR curves)
│       └── logger.py                     # Logging configuration
└── data/                                 # Place datasets here (see below)
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

We use the **Sirinam (AWF)** dataset for demonstration. The dataset contains network traffic traces from 95 monitored websites, with each trace stored as a `.cell` file.

**Data format:** Each `.cell` file contains two columns per line:
- Column 1: Timestamp (float)
- Column 2: Packet direction (+1 for outgoing, -1 for incoming)

**Directory structure:**
```
data/
├── 0-0.cell      # class 0, instance 0
├── 0-1.cell      # class 0, instance 1
├── ...
├── 94-999.cell   # class 94, instance 999
```

Place the dataset under `./data/sirinam/` or specify the path via `--data-path`.

## Usage

The attack pipeline consists of three steps:

### Step 1: Train Trigger (RL Optimization)

Train the LSTM policy network to find optimal trigger insertion positions:

```bash
python src/run_train_backdoor_trigger.py \
    --data-path ./data/sirinam/ \
    --verbose --one-fold \
    --model rf --feature-type tam --seq-length 1800 \
    --mode train \
    --lr0 0.000001 --epochs 30 --batch-size 1024 \
    --gpu 0 \
    --asr --return-backdoored \
    --backdoor-ratio 0.5 \
    --backdoor-lable 0 \
    --backdoor-label-type lc \
    --distance levenshtein \
    --trigger-train True \
    --backdoor-length 20000 \
    --backdoor-nums 7 \
    --eval-nums -1
```

The trained trigger will be saved under `./checkpoints/`.

### Step 2: Generate Trigger Patterns

Generate trigger patterns from the trained policy network. **We provide a pre-trained RL policy weight** (54KB) at `pretrained/trigger_policy.pth`, so you can skip Step 1 and start directly from here:

```bash
python src/run_test_backdoor_trigger.py \
    --data-path ./data/sirinam/ \
    --verbose --one-fold \
    --mon-classes 95 --page-per-class 1 \
    --mon-inst-train -1 --mon-inst 1000 \
    --model rf --feature-type tam --seq-length 1800 \
    --mode train \
    --lr0 0.000001 --epochs 1 --batch-size 512 \
    --gpu 0 \
    --pretrained ./pretrained/trigger_policy.pth \
    --trigger-pretrain True \
    --asr --return-backdoored \
    --backdoor-ratio 0.5 \
    --backdoor-lable 0 \
    --backdoor-label-type lc \
    --distance levenshtein \
    --trigger-train True \
    --backdoor-length 4000 \
    --backdoor-nums 7 \
    --eval-nums -1
```

### Step 3: Backdoor Attack Evaluation

Evaluate the backdoor attack on victim models in the **closed-world** setting:

```bash
# Example: Attack on RF model
python src/run_attack_backdoor.py \
    --data-path ./data/sirinam/ \
    --verbose --one-fold \
    --mon-classes 95 --page-per-class 1 \
    --mon-inst-train -1 --mon-inst 1000 \
    --model rf --feature-type tam --seq-length 1800 \
    --epochs 50 --lr0 0.0005 --weight-decay 0.001 --batch-size 200 \
    --gpu 0 \
    --asr --return-backdoored \
    --backdoor-ratio 1.0 \
    --backdoor-type BackdoorRLNet_optimize_multi_patch_in \
    --trigger-pth ./checkpoints/<your_trigger>/positions_counts_epoch1.json \
    --backdoor-length 20000 --backdoor-num 7 \
    --backdoor-label-type lc --backdoor-lable 33 \
    --eval-nums -1
```

For the **open-world** setting, add `--open-world --unmon-inst 40000`:

```bash
python src/run_attack_backdoor.py \
    --data-path ./data/sirinam/ \
    --verbose --one-fold \
    --mon-classes 95 --page-per-class 1 \
    --mon-inst-train -1 --mon-inst 1000 \
    --unmon-inst 40000 --open-world \
    --model rf --feature-type tam --seq-length 1800 \
    --epochs 50 --lr0 0.0005 --weight-decay 0.001 --batch-size 200 \
    --gpu 0 \
    --asr --return-backdoored \
    --backdoor-ratio 1.0 \
    --backdoor-type BackdoorRLNet_optimize_multi_patch_in \
    --trigger-pth ./checkpoints/<your_trigger>/positions_counts_epoch1.json \
    --backdoor-length 20000 --backdoor-num 7 \
    --backdoor-label-type lc --backdoor-lable 33 \
    --eval-nums -1
```

### Supported Victim Models

| Model | `--model` | `--feature-type` | `--seq-length` | Recommended hyperparams |
|-------|-----------|-------------------|----------------|------------------------|
| RF    | `rf`      | `tam`             | 1800           | `--lr0 0.0005 --weight-decay 0.001 --batch-size 200` |
| DF    | `df`      | `df`              | 10000          | `--lr0 0.002 --weight-decay 0 --batch-size 64` |
| Var-CNN | `varcnn` | `tiktok`         | 10000          | `--lr0 0.002 --weight-decay 0.0005 --batch-size 64` |
| ARES  | `ares`    | `df`              | 10000          | `--lr0 0.002 --weight-decay 0.001 --batch-size 256` |
| TMWF  | `tmwf`    | `df`              | 30720          | `--lr0 0.002 --weight-decay 0.0005 --batch-size 256` |
| Tiktok | `df`     | `tiktok`          | 10000          | `--lr0 0.002 --weight-decay 0 --batch-size 64` |
| K-FP  | `kfingerprinting` | -          | -              | Use `run_attack_backdoor_kf.py` |

### Backdoor Attack Types

| Method | `--backdoor-type` | Description |
|--------|-------------------|-------------|
| Ours (RL-optimized) | `BackdoorRLNet_optimize_multi_patch_in` | RL-based trigger with `--trigger-pth` |
| Ours + FAST | `FAST_optimize_BackdoorRLNet_multi_patch_in` | Combined RL + FAST optimization |
| BadNet (random) | `badnet_random_in` | Random trigger insertion |
| BadNet (patch) | `badnet_patch_in` | Fixed patch insertion |
| BadNet (multi-patch) | `badnet_multi_patch_in` | Multi-patch fixed insertion |
| FAST | `FAST_optimize_multi_patch_in` | FAST distance-optimized insertion |
| OSAD | `OSAD_optimize_multi_patch_in` | OSAD distance-optimized insertion |
| Damerau | `DAMERAU_optimize_multi_patch_in` | Damerau-Levenshtein optimized |

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--backdoor-ratio` | Poisoning ratio (fraction of training data poisoned) |
| `--backdoor-length` | Total length of trigger pattern |
| `--backdoor-num` / `--backdoor-nums` | Number of trigger patches |
| `--backdoor-lable` | Target label for the attack |
| `--backdoor-label-type` | Label type: `lc` (label-consistent) or `poi` (poisoned) |
| `--distance` | Distance metric for RL reward: `levenshtein`, `hamming`, `sequence` |
| `--trigger-pth` | Path to pre-trained trigger (JSON file) |
| `--open-world` | Enable open-world evaluation |
| `--eval-nums` | Number of evaluation samples (-1 for all) |

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{liang2026trapflow,
  title={Trapflow: Controllable website fingerprinting defense via dynamic backdoor learning},
  author={Liang, Siyuan and Gong, Jiajun and Fang, Tianmeng and Liu, Aishan and Wang, Tao and Cao, Xiaochun and Tao, Dacheng and Ee-Chien, Chang},
  journal={IEEE Transactions on Information Forensics and Security},
  year={2026},
  publisher={IEEE}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
