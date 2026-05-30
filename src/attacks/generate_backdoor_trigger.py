import argparse
import os
import time
from pathlib import Path
from typing import Tuple
import copy
import numpy as np
import torch
import logging
from pathlib import Path
import torch.nn as nn
from ignite.contrib.handlers.param_scheduler import LRScheduler
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.metrics import Loss, Accuracy
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torch.nn.functional as F
from attacks import Attack
from attacks.modules import DFNet, InceptionNet, TMWF_DFNet, RFNet, RFNet2, RFNet3, RFNet4
from attacks.attack_model import InsertPolicyNetwork
from utils.data import TestTriggerDataset
from utils.general import get_flist_label_multi_domain, increment_path, PR_THRES_NUM, get_grad_norm, select_fast_slow
from utils.metric import WFMetric, WFPRCurve, ASR
from attacks.function import TITCriterion, TITCriterion_v, sample_action, insert_backdoor, cosine_distance_per_sample, batch_hamming_distance_with_padding, batch_fast_levenshtein_like_distance, fast_levenshtein_like_distance, adjust_length, compute_reward
feature_transform_func = None
from attacks.train_backdoor_trigger import round_with_ste

import torch
import torch.nn as nn


class FinetuneTest(Attack):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        last_part = self.args.data_path.rstrip('/').split('/')[-1]
        self.dataset_name = last_part.split('_')[0]
        # 配置日志记录器，保存到文件
        if args.trigger_pretrain:
            self.checkpoint_path_rimmer = increment_path(
                Path(self.args.model_path) / "BackdoorRLNet_{}_{}_{}_{}_{}_{}_{}_{}".format(
                    'rimmer', self.args.distance, self.args.lr0, self.args.model, self.args.feature_type,
                    self.args.backdoor_type, str(self.args.backdoor_length), str(self.args.backdoor_nums)
                ),
                sep='_', exist_ok=self.args.exist_ok, mkdir=True)
        self.checkpoint_path = increment_path(
            Path(self.args.model_path) / "BackdoorRLNet_{}_{}_{}_{}_{}_{}_{}_{}".format(
                self.dataset_name, self.args.distance, self.args.lr0, self.args.model, self.args.feature_type,
                self.args.backdoor_type, str(self.args.backdoor_length), str(self.args.backdoor_nums)
            ),
            sep='_', exist_ok=self.args.exist_ok, mkdir=True)
        self.checkpoint_name = 'train_epoch_' + str(self.args.epochs) + '.pth'

        # 数据集配置
        self.nmc, self.flist, self.labels = get_flist_label_multi_domain(
            self.args.data_path,
            mon_cls=self.args.mon_classes,
            mon_inst=self.args.mon_inst,
            unmon_inst=self.args.unmon_inst,
            page_per_class=self.args.page_per_class,
            suffix=self.args.suffix
        )

    def _get_data(self, flist: np.ndarray, labels: np.ndarray, is_train: bool = True,
                  return_backdoored: bool = False, backdoor_lable: int = 0,
                  backdoor_type: str = 'default', include_filenames: bool = False) -> (TestTriggerDataset, DataLoader):
        batch_size = self.args.batch_size

        dataset = TestTriggerDataset(
            self.args, flist, labels, is_train,
            return_backdoored=return_backdoored,
            backdoor_lable=backdoor_lable,
            backdoor_type=backdoor_type,
            include_filenames=include_filenames,
        )
        print(f"Using {backdoor_type} backdoor attack strategy. Label attack is {self.args.backdoor_label_type}.")
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=is_train,
            num_workers=self.args.workers, collate_fn=TestTriggerDataset.collate_fn
        )
        return dataset, loader

    def generate_positions_and_counts(self):
        # 初始化模型
        model = InsertPolicyNetwork(hidden_dim=32).to(self.device)

        # 加载训练好的模型权重
        if self.args.pretrained:
            model_path = Path(self.args.pretrained)
        elif self.args.trigger_pretrain:
            model_path = self.checkpoint_path_rimmer / self.checkpoint_name
        else:
            model_path = self.checkpoint_path / self.checkpoint_name
        state_dict = torch.load(model_path, map_location=self.device)
        print("Keys in the checkpoint state_dict:")
        print(state_dict.keys())

        print("Keys in the model's state_dict:")
        print(model.state_dict().keys())
        
        model.load_state_dict(torch.load(model_path, map_location=self.device))

        # 设置模型为评估模式
        model.eval()

        # 创建包含所有数据的数据集和加载器
        dataset, loader = self._get_data(
            self.flist, self.labels, is_train=False,
            return_backdoored=False, backdoor_lable=0,
            backdoor_type='default', include_filenames=True
        )

        results = {}
        num_positions = self.args.backdoor_nums
        total_inserts = self.args.backdoor_length
        total_files = len(dataset)
        # 迭代数据加载器
        with torch.no_grad():
            total_processed = 0  # 初始化一个计数器来跟踪已处理的总文件数
            for batch in loader:
                xs, ys, filenames = batch  # 现在 batch 包含文件名
                for i, x in enumerate(xs):
                    x_tensor = torch.from_numpy(x).float().to(self.device)
                    seq_len = x_tensor.shape[0]

                    if seq_len < 2:
                        continue  # 跳过长度不足的序列

                    # 随机选择 num_positions 个插入位置
                    positions = torch.randint(1, seq_len, (num_positions,), device=self.device)

                    predicted_lengths_list = []

                    for idx in positions:
                        idx = idx.item()
                        # 获取对应位置的序列前缀
                        x_prefix = x_tensor[:idx, :]  # 形状为 [idx, input_dim]

                        if x_prefix.size(0) < 1:
                            continue  # 跳过长度不足的前缀

                        # 增加批次维度
                        x_prefix = x_prefix.unsqueeze(0)  # [1, idx, input_dim]

                        # 模型预测插入长度
                        logits = model(x_prefix)  # [1, 1]
                        predicted_length = F.softplus(logits.squeeze(0))  # 使用 Softplus 激活，确保非负

                        predicted_lengths_list.append(predicted_length)  # 列表元素为张量

                    if len(predicted_lengths_list) == 0:
                        continue  # 如果没有有效的预测，跳过

                    # 将预测的插入长度拼接成张量
                    predicted_lengths = torch.stack(predicted_lengths_list)  # [num_positions]

                    # 归一化处理，使总和为 total_inserts
                    lengths_sum = predicted_lengths.sum()
                    predicted_lengths = predicted_lengths / lengths_sum * total_inserts

                    predicted_lengths_continuous = predicted_lengths.squeeze(0)  # [num_positions]
                    predicted_lengths = round_with_ste(predicted_lengths_continuous)

                    if predicted_lengths.size(-1) == 1:
                        if predicted_lengths.size(0) != 1:
                            predicted_lengths = predicted_lengths.squeeze(-1)  # 将 [4, 1] 变为 [4]

                    # 检查 positions 和 predicted_lengths 的长度
                    assert positions.size(0) == num_positions, f"Expected positions length to be 4, but got {positions.size(0)}"
                    assert predicted_lengths.size(0) == num_positions, f"Expected predicted_lengths length to be 4, but got {predicted_lengths.size(0)}"

                    # 继续执行代码
                    filename = filenames[i]
                    positions_i = positions.cpu().tolist()
                    counts_i = predicted_lengths.cpu().tolist()

                    results[filename] = {
                        'positions': positions_i,
                        'counts': counts_i
                    }

                    # 打印当前处理的文件名和累计处理的文件数量
                    total_processed += 1
                    if total_processed % 1000 == 0:
                        print(f"Processed {total_processed} files in {total_files} files so far. Current file: {filename}")


        # 将结果保存到 JSON 文件
        import json
        json_name = 'positions_counts' + '_epoch'+ str(self.args.epochs)+ '.json'
        json_path = self.checkpoint_path / json_name
        with open(json_path, 'w') as f:
            json.dump(results, f)

        self.logger.info(f"Positions and counts saved at {json_path}")

    def run(self):
        self.generate_positions_and_counts()

