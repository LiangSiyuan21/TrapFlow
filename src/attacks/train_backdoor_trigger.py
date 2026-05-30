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
from utils.data import TrainTriggerDataset
from utils.general import get_flist_label_multi_domain, increment_path, PR_THRES_NUM, get_grad_norm, select_fast_slow
from utils.metric import WFMetric, WFPRCurve, ASR
from attacks.function import TITCriterion, TITCriterion_v, sample_action, insert_backdoor, cosine_distance_per_sample, batch_hamming_distance_with_padding, batch_fast_levenshtein_like_distance, fast_levenshtein_like_distance, adjust_length, compute_reward
feature_transform_func = None
from utils.compute_WF_distance import fast_levenshtein_like_distance_gpu, efficient_sequence_distance

import torch
import torch.nn as nn

def round_with_ste(x):
    """
    使用 Straight-Through Estimator 进行取整操作
    前向传播中取整，反向传播中梯度直接传递
    """
    y = torch.round(x)
    return x + (y - x).detach()


# def insert_backdoor(trace_tensor, topk_indices, backdoor_length, backdoor_nums, device):
#     """
#     在 trace_tensor 中插入 backdoor。

#     :param trace_tensor: torch.Tensor, 形状为 [length, 2]，代表原始 trace
#     :param topk_indices: torch.Tensor, 插入的索引位置，形状为 [k]
#     :param backdoor_length: int, backdoor 的总长度
#     :param backdoor_nums: int, 插入 backdoor 的数量
#     :return: 带有 backdoor 的 trace_tensor
#     """
#     num_patches = backdoor_nums
#     patch_length = int(backdoor_length / num_patches)
    
#     # 创建 trace_tensor 的副本，以免修改原始 trace
#     trace_bd = trace_tensor.clone()
#     print(f"trace_tensor grad_fn before insert: {trace_bd.grad_fn}")
#     valid_patches = 0

#     # 遍历 topk_indices 来插入 backdoor
#     for idx in topk_indices:
#         idx = idx.item()  # 将 Tensor 转换为 Python 的整数

#         if idx > 0 and idx < len(trace_bd):
#             # 计算插入的 backdoor 数据（时间均匀分布，值为 -1）
#             pattern_in = torch.linspace(trace_bd[idx-1, 0], trace_bd[idx, 0], min(patch_length, len(trace_bd) - idx))
#             pattern_in_2d = torch.stack((pattern_in, -torch.ones_like(pattern_in)), dim=-1).to(device)  # 创建 [时间, -1] 的 2D 模式
            
#             # 在 trace_bd 中插入 backdoor 数据
#             trace_bd = torch.cat((trace_bd, pattern_in_2d), dim=0)
#             valid_patches += 1

#     # 确保 trace_bd 按时间排序，基于第 0 列（时间）
#     trace_bd = trace_bd[trace_bd[:, 0].argsort()]
#     print(f"trace_tensor grad_fn before insert: {trace_bd.grad_fn}")

#     return trace_bd


def gumbel_softmax_sample(logits, temperature):
    noise = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(noise + 1e-20) + 1e-20)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)

class FinetuneTest(Attack):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        
        last_part = self.args.data_path.rstrip('/').split('/')[-1]
        self.dataset_name = last_part.split('_')[0]
        # Configure logger to save to a file
        log_path = increment_path(
                Path(self.args.model_path) / "BackdoorRLNet_{}_{}_{}_{}_{}_{}_{}_{}".format(self.dataset_name, self.args.distance,self.args.lr0, self.args.model, self.args.feature_type, self.args.backdoor_type, str(self.args.backdoor_length), str(self.args.backdoor_nums)),
                sep='_', exist_ok=self.args.exist_ok, mkdir=True)
        log_dir = Path(log_path) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)  # Create log directory if it doesn't exist
        log_file = log_dir / "training.log"
        
        # Clear previous handlers if they exist
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Create file handler which logs even debug messages
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add the handlers to the logger
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        
        self.logger.info("Logger configured to write to file: {}".format(log_file))
        
        # Your existing initialization code
        assert self.args.aug_times >= 0, "Augmentation times must be non-negative"
        assert self.args.averaging_times > 0, "Averaging times must be non-negative"
        assert self.args.page_per_class > 0, "Page per class must be non-negative"

        self.args.unmon_inst = args.unmon_inst if args.open_world else 0

        # dataset configs
        self.nmc, self.flist, self.labels = get_flist_label_multi_domain(self.args.data_path,
                                                                         mon_cls=self.args.mon_classes,
                                                                         mon_inst=self.args.mon_inst,
                                                                         unmon_inst=self.args.unmon_inst,
                                                                         page_per_class=self.args.page_per_class,
                                                                         suffix=self.args.suffix)

        self.nc = len(np.unique(self.labels))
        self.unmon_inst = args.unmon_inst if args.open_world else 0

        assert self.nc == self.nmc + int(self.args.open_world), \
            "Number of classes does not match the expected number of classes"

        self.logger.info("Number of data: {}, Num of classes: {} + {}".format(len(self.flist), self.nmc,
                                                                              self.nc - self.nmc))

        self.feature_type = self.args.feature_type
        self.aug_times = args.aug_times

        self.amp_mode = 'amp' if (not self.args.not_amp) and self.device != torch.device("cpu") else None

        self.logger.info('Augmentation times: {} | Averaging times: {}'.format(
            self.args.aug_times, self.args.averaging_times))

        if not self.args.nosave and self.args.mode == 'train':
            self.checkpoint_path = increment_path(
                Path(self.args.model_path) / "BackdoorRLNet_{}_{}_{}_{}_{}_{}_{}_{}".format(self.dataset_name, self.args.distance,self.args.lr0, self.args.model, self.args.feature_type, self.args.backdoor_type, str(self.args.backdoor_length), str(self.args.backdoor_nums)),
                sep='_', exist_ok=self.args.exist_ok, mkdir=True)

    def _build_model(self):
        if self.args.verbose:
            self.logger.info("Building TriggerInsertionTransformer_model: {} | Feature: {}".format(self.args.model, self.args.feature_type))

        ch = 2 if self.feature_type == 'tam' or self.feature_type == 'tam+' else 1

        if self.args.model == 'df':
            TriggerInsertionTransformer_model = DFNet(length=self.args.seq_length, num_classes=self.nc, in_channels=ch)

        elif self.args.model == 'inception':
            if self.feature_type == 'tam' or self.feature_type == 'tam+':
                TriggerInsertionTransformer_model = InceptionNet(length=self.args.seq_length, num_classes=self.nc, in_channels=1,
                                     num_kernels=self.args.num_kernels)
            else:
                TriggerInsertionTransformer_model = InceptionNet(length=self.args.seq_length, num_classes=self.nc,
                                     in_channels=self.args.fusion_granularity,
                                     num_kernels=self.args.num_kernels)

        elif self.args.model == 'tmwf':
            TriggerInsertionTransformer_model = TMWF_DFNet(length=self.args.seq_length, num_classes=self.nc, in_channels=ch)

        elif self.args.model == 'rf':
            assert self.feature_type == 'tam' or self.feature_type == 'tam+' or self.feature_type == 'fusion', \
                "RF2 only supports TAM or fusion features"
            if self.feature_type == 'tam' or self.feature_type == 'tam+':
                TriggerInsertionTransformer_model = RFNet(num_classes=self.nc)
            else:
                TriggerInsertionTransformer_model = RFNet(num_classes=self.nc, in_channel=self.args.fusion_granularity + 1)

        elif self.args.model == 'rf2':
            assert self.feature_type == 'tam' or self.feature_type == 'tam+' or self.feature_type == 'fusion', \
                "RF2 only supports TAM or fusion features"
            if self.feature_type == 'tam' or self.feature_type == 'tam+':
                TriggerInsertionTransformer_model = RFNet2(num_classes=self.nc, in_channel=1)
            else:
                TriggerInsertionTransformer_model = RFNet2(num_classes=self.nc, in_channel=self.args.fusion_granularity)

        elif self.args.model == 'rf3':
            assert self.feature_type == 'tam' or self.feature_type == 'tam+' or self.feature_type == 'fusion', \
                "RF3 only supports TAM or fusion features"
            if self.feature_type == 'tam' or self.feature_type == 'tam+':
                TriggerInsertionTransformer_model = RFNet3(num_classes=self.nc, in_channel=1)
            else:
                TriggerInsertionTransformer_model = RFNet3(num_classes=self.nc, in_channel=self.args.fusion_granularity)
        elif self.args.model == 'rf4':
            assert self.feature_type == 'tam' or self.feature_type == 'tam+' or self.feature_type == 'fusion', \
                "RF3 only supports TAM or fusion features"
            if self.feature_type == 'tam' or self.feature_type == 'tam+':
                TriggerInsertionTransformer_model = RFNet4(num_classes=self.nc, in_channel=1)
            else:
                TriggerInsertionTransformer_model = RFNet4(num_classes=self.nc, in_channel=self.args.fusion_granularity,
                               num_kernels_1d=self.args.num_kernels, num_kernels_2d=self.args.num_kernels)

        else:
            raise NotImplementedError("Model {} is not implemented.".format(self.args.model))
        return TriggerInsertionTransformer_model.to(self.device)

    def _get_data(self, flist: np.ndarray, labels: np.ndarray, is_train: bool = True,
                return_backdoored: bool = False ,backdoor_lable: int = 0, backdoor_type: str = 'default') -> (TrainTriggerDataset, DataLoader):
        batch_size = self.args.batch_size

        dataset = TrainTriggerDataset(self.args, flist, labels, is_train, return_backdoored=return_backdoored, backdoor_lable=backdoor_lable, backdoor_type=backdoor_type)
        print(f"Using {backdoor_type} backdoor attack strategy. label attack is {self.args.backdoor_label_type}.")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train,
                            num_workers=self.args.workers, collate_fn=TrainTriggerDataset.collate_fn)
        return dataset, loader

    # @timeit
    def run(self, one_fold_only: bool = False):
        if self.args.open_world:
            res = np.zeros((PR_THRES_NUM, 5))  # tps, fps, wps, fns, tns
        else:
            res = np.zeros(4)  # tp, fp, p, n (placeholder, no negative actually)

        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=self.args.seed)

        # if self.args.mon_inst_train > 0:
        #     train_total_num = self.args.mon_inst_train * self.nmc + self.unmon_inst
        #     self.logger.info("Partially use {} + {} instances for training".format(
        #         self.args.mon_inst_train * self.nmc, self.unmon_inst))
        # else:
        #     train_total_num = -1

        for fold, (train_index, test_index) in enumerate(sss.split(self.flist, self.labels)):
            if one_fold_only and fold > 0:
                break
            _train_list, _train_labels = self.flist[train_index], self.labels[train_index]
            
            # split the training set into training and validation set
            train_list, val_list, train_labels, val_labels = train_test_split(_train_list, _train_labels,
            test_size=0.10,
            random_state=self.args.seed,
            stratify=_train_labels)

            if len(val_list) == len(val_labels) and len(val_list) > self.args.eval_nums:
                val_list = val_list[:self.args.eval_nums]
                val_labels = val_labels[:self.args.eval_nums]

            # adjust mon inst train number and unmon inst train number
            train_list_new, train_labels_new = [], []
            if self.args.mon_inst_train > 0:
                self.logger.info("Use {} instances per class for training".format(self.args.mon_inst_train))
                for lb in range(self.nmc):
                    idx = np.where(train_labels == lb)[0]
                    total_num = min(len(idx), self.args.mon_inst_train)
                    idx = np.random.choice(idx, total_num, replace=False)
                    train_list_new.extend(train_list[idx])
                    train_labels_new.extend(train_labels[idx])
            else:
                train_list_new.extend(train_list[train_labels < self.nmc])
                train_labels_new.extend(train_labels[train_labels < self.nmc])

            if self.args.open_world:
                if self.args.unmon_inst_train > 0:
                    self.logger.info("Use {} unmonitored instances for training".format(self.args.unmon_inst_train))
                    idx = np.where(train_labels == self.nmc)[0]
                    total_num = min(len(idx), self.args.unmon_inst_train)
                    idx = np.random.choice(idx, total_num, replace=False)
                    train_list_new.extend(train_list[idx])
                    train_labels_new.extend(train_labels[idx])
                else:
                    train_list_new.extend(train_list[train_labels == self.nmc])
                    train_labels_new.extend(train_labels[train_labels == self.nmc])

            train_list_new = np.array(train_list_new)
            train_labels_new = np.array(train_labels_new)
            if len(test_index) > self.args.eval_nums:
                test_index = test_index[:self.args.eval_nums]
            test_list, test_labels = self.flist[test_index], self.labels[test_index]
            res_one_fold = self.train(fold + 1, train_list_new, train_labels_new, val_list, val_labels, test_list,
                                      test_labels, self.args.backdoor_lable, self.args.backdoor_type)
            self.logger.info("-" * 10)
            self.logger.info("finished")

    def train_step(self, engine: Engine, batch: Tuple, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device, scaler: torch.cuda.amp.GradScaler, clip_value: float, use_amp: bool):
        model.train()
        optimizer.zero_grad()

        xs, ys, _ = batch
        total_loss = 0.0
        num_positions = self.args.backdoor_nums
        total_inserts = self.args.backdoor_length

        for x in xs:
            x_tensor = torch.from_numpy(x).float().to(device)
            seq_len = x_tensor.shape[0]

            if seq_len < 2:
                continue  # 跳过长度不足的序列

            # 随机选择 num_positions 个插入位置
            positions = torch.randint(1, seq_len, (num_positions,), device=device)

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

            # 使用 STE 进行取整
            predicted_lengths = round_with_ste(predicted_lengths_continuous)  # [num_positions]

            # 添加总和约束损失
            sum_loss = ((predicted_lengths.sum() - total_inserts) ** 2)

            # 插入操作
            new_traces = [x_tensor]

            for idx_tensor, patch_length in zip(positions, predicted_lengths):
                idx = idx_tensor.item()

                if idx >= seq_len:
                    continue  # 防止索引越界

                # 将 patch_length 转换为整数
                insert_length = int(patch_length.item())

                if insert_length > 0:
                    # 生成插入时间戳
                    pattern_in_time = torch.linspace(
                        x_tensor[idx - 1, 0],
                        x_tensor[idx, 0],
                        steps=insert_length,
                        device=device
                    )
                    # 插入的值为 -1
                    pattern_in_value = torch.full((insert_length,), -1.0, device=device)
                    pattern_in_2d = torch.stack((pattern_in_time, pattern_in_value), dim=1)
                    new_traces.append(pattern_in_2d)

            # 合并新插入的 traces
            new_trace = torch.cat(new_traces, dim=0)
            sorted_indices = torch.argsort(new_trace[:, 0])
            sorted_trace = new_trace[sorted_indices]

            # 计算序列差异损失
            # 截断或填充 sorted_trace，使其与 x_tensor 长度一致
            target_length = x_tensor.size(0)
            if sorted_trace.size(0) > target_length:
                sorted_trace = sorted_trace[:target_length]
            elif sorted_trace.size(0) < target_length:
                padding = torch.zeros(target_length - sorted_trace.size(0), sorted_trace.size(1), device=device)
                sorted_trace = torch.cat([sorted_trace, padding], dim=0)

            if self.args.distance == 'levenshtein':
                sequence_loss = -fast_levenshtein_like_distance_gpu(sorted_trace, x_tensor)
            elif self.args.distance == 'sequence':
                sequence_loss = -efficient_sequence_distance(sorted_trace, x_tensor)
            elif self.args.distance == 'hamming':
                sequence_loss = -batch_hamming_distance_with_padding(sorted_trace.unsqueeze(0), x_tensor.unsqueeze(0))
                

            # 总损失
            lambda_weight = 0.001  # 根据需要调整权重
            loss = sequence_loss + sum_loss * lambda_weight

            total_loss += loss

        if total_loss == 0.0:
            return 0.0  # 如果没有有效的序列，返回零损失

        total_loss = total_loss / len(xs)
        print(total_loss)
        # 反向传播和优化
        if use_amp:
            with torch.cuda.amp.autocast():
                total_loss = total_loss

            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

        return total_loss.item()





    def create_supervised_trainer(self, TriggerInsertionTransformer_model: nn.Module, TIT_optimizer: torch.optim, criterion: nn.Module,
                                  device: torch.device = None, clip_value: float = 1.0,
                                  use_amp: bool = False) -> Engine:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        return Engine(lambda engine, batch: self.train_step(engine, batch, TriggerInsertionTransformer_model, TIT_optimizer,
                                                            criterion, device, scaler,
                                                            clip_value, use_amp))

    def test_step(self, engine, batch, TriggerInsertionTransformer_model, device):
        TriggerInsertionTransformer_model.eval()
        with torch.no_grad():
            x, y, _ = batch
            x_tensors = [torch.from_numpy(adjust_length(trace)).float() for trace in x]
            x_batch = torch.stack(x_tensors).to(device)
            y_batch = torch.tensor(y, dtype=torch.long, device=device)

            num_positions = self.args.backdoor_num
            total_inserts = self.args.backdoor_length

            trace_bd_list = []

            for x_tensor in x_tensors:
                x_tensor = x_tensor.to(device)
                seq_len = x_tensor.size(0)

                if seq_len < 2:
                    continue  # 跳过长度不足的序列

                # 随机选择 num_positions 个插入位置
                positions = torch.randint(1, seq_len, (num_positions,), device=device)

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
                    logits = TriggerInsertionTransformer_model(x_prefix)  # [1, 1]
                    predicted_length = F.softplus(logits.squeeze(1))  # 使用 Softplus 激活，确保非负

                    predicted_lengths_list.append(predicted_length)  # 列表元素为张量

                if len(predicted_lengths_list) == 0:
                    continue  # 如果没有有效的预测，跳过

                # 将预测的插入长度拼接成张量
                predicted_lengths = torch.stack(predicted_lengths_list)  # [num_positions, 1]
                predicted_lengths = predicted_lengths.squeeze(1)  # 转换为 [num_positions]

                # 归一化处理，使总和为 total_inserts
                lengths_sum = predicted_lengths.sum()
                predicted_lengths = predicted_lengths / lengths_sum * total_inserts

                # 插入操作
                insert_lengths = predicted_lengths.round().long()  # 转换为整数插入长度

                # 调整插入长度总和，使其等于 total_inserts
                difference = total_inserts - insert_lengths.sum().item()
                while difference != 0:
                    for i in range(len(insert_lengths)):
                        if difference == 0:
                            break
                        if difference > 0:
                            insert_lengths[i] += 1
                            difference -= 1
                        else:
                            if insert_lengths[i] > 0:
                                insert_lengths[i] -= 1
                                difference += 1

                # 插入操作
                new_traces = [x_tensor]

                for idx_tensor, insert_length in zip(positions, insert_lengths):
                    idx = idx_tensor.item()
                    insert_length = insert_length.item()

                    if idx >= seq_len:
                        continue  # 防止索引越界

                    if insert_length > 0:
                        # 生成插入时间戳
                        pattern_in_time = torch.linspace(
                            x_tensor[idx - 1, 0],
                            x_tensor[idx, 0],
                            steps=insert_length,
                            device=device
                        )
                        # 插入的值为 -1
                        pattern_in_value = torch.full((insert_length,), -1.0, device=device)
                        pattern_in_2d = torch.stack((pattern_in_time, pattern_in_value), dim=1)
                        new_traces.append(pattern_in_2d)

                # 合并新插入的 traces
                new_trace = torch.cat(new_traces, dim=0)
                sorted_indices = torch.argsort(new_trace[:, 0])
                sorted_trace = new_trace[sorted_indices]

                trace_bd_list.append(sorted_trace)

            # 截断或填充 trace_bd_list，使其长度一致
            trace_bd_list_truncated = [trace[:10000, :] if trace.size(0) > 10000 else trace for trace in trace_bd_list]
            trace_bd_tensor = torch.stack(trace_bd_list_truncated)

            # 计算平均距离
            x_tensors = torch.stack(x_tensors)
            distances = batch_fast_levenshtein_like_distance(x_tensors, trace_bd_tensor)
            average_distance = sum(distances) / len(distances)

        return average_distance


    def create_supervised_evaluator(self, TriggerInsertionTransformer_model, device):
        def _inference(engine, batch):
            # Here, we use self.test_step
            return self.test_step(engine, batch, TriggerInsertionTransformer_model, device)
        
        # Engine is an Ignite construct that processes a given batch of data by calling the provided _inference function
        evaluator = Engine(_inference)
        return evaluator

    def train(self, fold: int, train_list: np.ndarray, train_labels: np.ndarray,
          val_list: np.ndarray, val_labels: np.ndarray,
          test_list: np.ndarray, test_labels: np.ndarray, backdoor_lable: int=0, backdoor_type: str = 'default') -> np.ndarray:
        total_change = 0.0
        train_dataset, train_loader = self._get_data(train_list, train_labels, is_train=True, return_backdoored=False, backdoor_lable=backdoor_lable, backdoor_type=backdoor_type)
        _, val_loader = self._get_data(val_list, val_labels, is_train=False, return_backdoored=False, backdoor_lable=backdoor_lable, backdoor_type=backdoor_type)
        _, test_loader = self._get_data(test_list, test_labels, is_train=False, return_backdoored=False, backdoor_lable=backdoor_lable, backdoor_type=backdoor_type)

        global feature_transform_func
        feature_transform_func = train_dataset.feature_transform_func

        TriggerInsertionTransformer_model = InsertPolicyNetwork(hidden_dim=32).to(self.device)
        TIT_criterion = TITCriterion_v(cost_id=1, cost_trans=0.01)

        criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)

        if self.args.mode == 'test':
            # load trained TriggerInsertionTransformer_model
            TriggerInsertionTransformer_model.load_state_dict(torch.load(self.args.model_path, map_location='cpu'))
            res = self.test(TriggerInsertionTransformer_model, test_loader, backdoored_test_loader, criterion, backdoor_lable)
        else:
            lr0 = self.args.lr0
            self.logger.info(f"Initial learning rate: {lr0}")

            TIT_optimizer = torch.optim.Adam(TriggerInsertionTransformer_model.parameters(), lr=lr0, weight_decay=self.args.weight_decay)

            step_scheduler = LambdaLR(TIT_optimizer, lr_lambda=lambda epoch: 0.2 ** (epoch / self.args.epochs))
            lr_scheduler = LRScheduler(step_scheduler)

            trainer = self.create_supervised_trainer(TriggerInsertionTransformer_model, TIT_optimizer, TIT_criterion, self.device, clip_value=5, use_amp=not self.args.not_amp)
            test_evaluator = self.create_supervised_evaluator(TriggerInsertionTransformer_model=TriggerInsertionTransformer_model, device=self.device)


            @trainer.on(Events.EPOCH_COMPLETED)
            def log_training_loss(engine: Engine):
                if self.args.verbose:
                    grad_norm = get_grad_norm(TriggerInsertionTransformer_model)

                    self.logger.info(f"Fold[{fold}] | Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] | "
                                     f"Loss: {engine.state.output:.2f} | "
                                     f"Norm: {grad_norm:.4f}")

            @trainer.on(Events.EPOCH_COMPLETED)
            def print_lr():
                if self.args.verbose:
                    self.logger.info(f"Current learning rate: {TIT_optimizer.param_groups[0]['lr']}")

            # @trainer.on(Events.EPOCH_COMPLETED)
            def log_test_results(engine):
                state = test_evaluator.run(test_loader)
                distance = state.output
                self.logger.info(f"Epoch {engine.state.epoch} - Average Distance: {distance:.4f}")

            @trainer.on(Events.EPOCH_COMPLETED)
            def save_model(engine: Engine):
                if not self.args.nosave and self.args.mode == 'train':
                    epoch = engine.state.epoch
                    model_path = self.checkpoint_path / f'train_epoch_{epoch}.pth'
                    torch.save(TriggerInsertionTransformer_model.state_dict(), model_path)
                    if self.args.verbose:
                        self.logger.info(f"Model saved at {model_path}")
            trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)
            trainer.run(train_loader, max_epochs=self.args.epochs)

        res = None
        torch.cuda.empty_cache()
        return res

