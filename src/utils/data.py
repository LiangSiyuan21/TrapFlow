import argparse
from functools import partial
import time
import numpy as np
import torch
import torch.utils.data as Data
from utils.backdoor_attack_strategy import BackdoorAttackStrategy, DefaultBackdoorAttackStrategy, AnotherBackdoorAttackStrategy, BadNetRandomStrategyOnlyIn, BadNetPatchStrategyOnlyIn, BadNetMultiPatchStrategyOnlyIn, OSADOptimizeMultiPatchStrategyOnlyIn, DAMERAUOptimizeMultiPatchStrategyOnlyIn, FASTOptimizeMultiPatchStrategyOnlyIn, BackdoorRLNetOptimizeMultiPatchStrategyOnlyIn, FASTOptimizeMultiPatchStrategyOnlyInTestFree,FASTOptimizeBackdoorRLNetMultiPatchStrategyOnlyIn, BadNetBackdoorRLNetMultiPatchStrategyOnlyIn, BadNetRandomBackdoorRLNetMultiPatchStrategyOnlyIn, SHAPOptimizeMultiPatchStrategyOnlyIn, TrojFlowStrategy

from attacks.function import adjust_length
from utils.augment import TrivialAugment
from utils.general import parse_trace, parse_trace_trafficsliver, feature_transform

import numpy as np

def random_remove(trace, backdoor_length, backdoor_num):
    n = trace.shape[0]
    if n <= 1:  # 如果trace长度小于等于1，则不进行任何操作
        return trace

    remove_pattern = int(backdoor_length / backdoor_num)
    
    # 计算可以移除的次数，保证trace长度始终大于1
    if n - remove_pattern < 2:
        return trace
    max_removals = min(backdoor_num, (n - 1) // remove_pattern)
    
    indices_to_remove = np.random.choice(np.arange(1, n - remove_pattern), max_removals, replace=False)
    for idx in sorted(indices_to_remove, reverse=True):
        temp_trace = np.delete(trace, slice(idx, idx + remove_pattern), axis=0)
        if temp_trace.shape[0] > 1:  # 确保移除后trace长度大于1
            trace = temp_trace
        else:  # 如果移除导致trace长度小于等于1，停止移除并保留原始状态
            break

    return trace


class SingleDomainDataset(Data.Dataset):
    def __init__(self, args: argparse.Namespace, flist: np.ndarray, labels: np.ndarray, is_train: bool):
        self.args = args
        self.flist = flist
        self.labels = labels
        self.feature_type = self.args.feature_type
        self.is_train = is_train

        self.aug_times = args.aug_times if is_train else 0
        self.averaging_times = args.averaging_times if is_train else 0

        self.feature_transform_func = partial(feature_transform, feature_type=self.feature_type,
                                              seq_length=self.args.seq_length, n_tam=self.args.n_tam,
                                              granularity=self.args.fusion_granularity)
        self.augmentor = TrivialAugment(self.args, self.flist, self.labels,
                                        feature_transform_func=self.feature_transform_func,
                                        averaging_times=self.averaging_times)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        sample_path = self.flist[idx]
        trace = parse_trace(sample_path)
        x1 = self.feature_transform_func(trace)
        x1 = torch.from_numpy(x1).float()

        if self.aug_times <= 0 and self.averaging_times <= 0:
            x = x1.reshape(1, *x1.shape)
        else:
            x = [x1]
            for _ in range(self.aug_times):
                x2 = self.augmentor(idx, trace)
                x2 = torch.from_numpy(x2).float()
                x.append(x2)
            x = torch.stack(x, dim=0)
        y = self.labels[idx]
        y = torch.tensor(y).long()
        # duplicate y according to x's first dimension
        y = y.repeat(x.shape[0])

        return x, y

    def __len__(self):
        return len(self.flist)

    @staticmethod
    def collate_fn(batch):
        x, y = zip(*batch)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        return x, y


class TrafficSliverDataset(SingleDomainDataset):
    def __init__(self, args: argparse.Namespace, flist: np.ndarray, labels: np.ndarray, is_train: bool):
        """
        Dataset for TrafficSliver defense. The train set will return all the sub-traces of a trace, while the
        validation/test set will only return one sub-trace randomly drawn from this sample
        """
        super().__init__(args, flist, labels, is_train)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        sample_path = self.flist[idx]
        traces = parse_trace_trafficsliver(sample_path)

        y = self.labels[idx]
        y = torch.tensor(y).long()

        if not self.is_train:
            # randomly draw a sub-trace from traces
            idx = np.random.choice(len(traces))
            trace = traces[idx]
            x = self.feature_transform_func(trace)
            x = torch.from_numpy(x).float()
            x = x.reshape(1, *x.shape)
            return x, y.reshape(1)

        # is_train
        xs = []
        ys = []
        for trace in traces:
            x = self.feature_transform_func(trace)
            x = torch.from_numpy(x).float()
            xs.append(x)
            ys.append(y)

        xs = torch.stack(xs, dim=0)
        ys = torch.tensor(ys).long()

        return xs, ys

    def __len__(self):
        return len(self.flist)

    @staticmethod
    def collate_fn(batch):
        x, y = zip(*batch)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        return x, y


class TuneDataset(Data.Dataset):
    def __init__(self, args: argparse.Namespace, flist: np.ndarray, labels: np.ndarray, fusion_granularity: int,
                 is_train: bool):
        self.args = args
        self.flist = flist
        self.labels = labels
        self.feature_type = self.args.feature_type
        self.is_train = is_train

        self.aug_times = args.aug_times if is_train else 0
        self.averaging_times = args.averaging_times if is_train else 0

        self.feature_transform_func = partial(feature_transform, feature_type=self.feature_type,
                                              seq_length=self.args.seq_length, n_tam=self.args.n_tam,
                                              granularity=fusion_granularity)
        self.augmentor = TrivialAugment(self.args, self.flist, self.labels,
                                        feature_transform_func=self.feature_transform_func,
                                        averaging_times=self.averaging_times)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        sample_path = self.flist[idx]
        trace = parse_trace(sample_path)
        x1 = self.feature_transform_func(trace)
        x1 = torch.from_numpy(x1).float()

        if self.aug_times <= 0 and self.averaging_times <= 0:
            x = x1.reshape(1, *x1.shape)
        else:
            x = [x1]
            for _ in range(self.aug_times):
                x2 = self.augmentor(idx, trace)
                x2 = torch.from_numpy(x2).float()
                x.append(x2)
            x = torch.stack(x, dim=0)
        y = self.labels[idx]
        y = torch.tensor(y).long()
        # duplicate y according to x's first dimension
        y = y.repeat(x.shape[0])

        return x, y

    def __len__(self):
        return len(self.flist)

    @staticmethod
    def collate_fn(batch):
        x, y = zip(*batch)
        x = torch.cat(x, dim=0)
        y = torch.cat(y, dim=0)
        return x, y


class CompareDataset(SingleDomainDataset):
    def __init__(self, args: argparse.Namespace, flist: np.ndarray, labels: np.ndarray, is_train: bool):
        super().__init__(args, flist, labels, is_train)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor):
        sample_path = self.flist[idx]
        trace = parse_trace(sample_path)
        x1 = self.feature_transform_func(trace)
        x1 = torch.from_numpy(x1).float()

        if not self.is_train:
            x = x1.reshape(1, *x1.shape)
        else:
            x = []
            for _ in range(2):
                # create two aug views of x
                x2 = self.augmentor(idx, trace)
                x2 = torch.from_numpy(x2).float()
                x.append(x2)
            x = torch.stack(x, dim=0)
        y = self.labels[idx]
        y = torch.tensor(y).long()

        return x, y

    def __len__(self):
        return len(self.flist)


class BackdoorDataset(SingleDomainDataset):
    def __init__(self, args: argparse.Namespace, flist: np.ndarray, labels: np.ndarray, is_train: bool, 
                 return_backdoored: bool = False, backdoor_lable: int = 0, backdoor_type: str = 'default'):
        super().__init__(args, flist, labels, is_train)
        self.args = args
        self.backdoor_ratio = args.backdoor_ratio
        self.return_backdoored = False if is_train else return_backdoored
        self.label_max = np.max(labels)
        self.label_min = np.min(labels)
        self.backdoor_lable = backdoor_lable
        self.backdoor_label_type = args.backdoor_label_type
        self.backdoor_length = args.backdoor_length

        # 根据 args.backdoor_type 选择攻击策略
        self.attack_strategy = self.get_attack_strategy(args.backdoor_type)

    def get_attack_strategy(self, backdoor_type: str) -> BackdoorAttackStrategy:
        strategies = {
            'default': DefaultBackdoorAttackStrategy,
            'another': AnotherBackdoorAttackStrategy,
            'badnet_random_in': BadNetRandomStrategyOnlyIn,
            'badnet_patch_in': BadNetPatchStrategyOnlyIn,
            'badnet_multi_patch_in': BadNetMultiPatchStrategyOnlyIn,
            'OSAD_optimize_multi_patch_in': OSADOptimizeMultiPatchStrategyOnlyIn,
            'DAMERAU_optimize_multi_patch_in': DAMERAUOptimizeMultiPatchStrategyOnlyIn,
            'FAST_optimize_multi_patch_in': FASTOptimizeMultiPatchStrategyOnlyIn,
            'BackdoorRLNet_optimize_multi_patch_in': BackdoorRLNetOptimizeMultiPatchStrategyOnlyIn,
            'FAST_optimize_multi_patch_in_test_free': FASTOptimizeMultiPatchStrategyOnlyInTestFree,
            'FAST_optimize_BackdoorRLNet_multi_patch_in':FASTOptimizeBackdoorRLNetMultiPatchStrategyOnlyIn,
            'badnet_BackdoorRLNet_optimize_multi_patch_in':BadNetBackdoorRLNetMultiPatchStrategyOnlyIn,
            'badnet_random_BackdoorRLNet_optimize_multi_patch_in':BadNetRandomBackdoorRLNetMultiPatchStrategyOnlyIn,
            'SHAP_optimize_multi_patch_in': SHAPOptimizeMultiPatchStrategyOnlyIn,
            'TrojanFlow': TrojFlowStrategy,
            # 可以在这里添加更多策略
        }
        return strategies.get(backdoor_type, DefaultBackdoorAttackStrategy)()

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, float):
        sample_path = self.flist[idx]
        trace = parse_trace(sample_path)
        original_trace = trace.copy()

        y = self.labels[idx]
        change_in_trace = 0.0
        if self.is_train:
            # train with backdoor_ratio, perturb the trace and change to the target label
            if self.args.adversarial_state == True:
                trace = trace
                y = y
            elif self.backdoor_label_type == 'poi':
                if y != self.backdoor_lable and np.random.rand() < self.backdoor_ratio:
                    if self.args.backdoor_type == 'BackdoorRLNet_optimize_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, sample_path)
                    elif self.args.backdoor_type == 'FAST_optimize_multi_patch_in_test_free':
                        trace = self.attack_strategy.perturb(self.args, trace, train_flog=self.is_train)
                    elif self.args.backdoor_type == 'FAST_optimize_BackdoorRLNet_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, path=None, train_flog=self.is_train)
                    elif self.args.backdoor_type == 'badnet_BackdoorRLNet_optimize_multi_patch_in' or self.args.backdoor_type == 'badnet_random_BackdoorRLNet_optimize_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, path=None, train_flog=self.is_train)
                    elif self.args.backdoor_type == 'TrojanFlow':
                        trace = self.attack_strategy.perturb(self.args, trace)
                    else:
                        trace = self.attack_strategy.perturb(self.args, trace)
                    change_in_trace = self.backdoor_length / len(original_trace)
                    y = self.backdoor_lable
            elif self.backdoor_label_type == 'lc':
                if y == self.backdoor_lable and np.random.rand() < self.backdoor_ratio:
                    if self.args.backdoor_type == "badnet_random_in":
                        trace = self.attack_strategy.perturb(self.args, trace)
                    elif self.args.backdoor_type == 'BackdoorRLNet_optimize_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, sample_path)
                    elif self.args.backdoor_type == 'FAST_optimize_multi_patch_in_test_free':
                        trace = self.attack_strategy.perturb(self.args, trace, train_flog=self.is_train)
                    elif self.args.backdoor_type == 'FAST_optimize_BackdoorRLNet_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, path=None, train_flog=self.is_train)
                    elif self.args.backdoor_type == 'badnet_BackdoorRLNet_optimize_multi_patch_in' or self.args.backdoor_type == 'badnet_random_BackdoorRLNet_optimize_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, path=None, train_flog=self.is_train)
                    elif self.args.backdoor_type == 'SHAP_optimize_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace)
                    elif self.args.backdoor_type == 'TrojanFlow':
                        trace = self.attack_strategy.perturb(self.args, trace)
                    else:
                        trace = trace
                    change_in_trace = self.backdoor_length / len(original_trace)
                    y = y
        else:
            # test
            if self.return_backdoored:
                start_time = time.time()
                if self.args.backdoor_type == "badnet_random_in":
                    trace = self.attack_strategy.perturb(self.args, trace)
                elif self.args.backdoor_type == 'BackdoorRLNet_optimize_multi_patch_in':
                    trace = self.attack_strategy.perturb(self.args, trace, path=sample_path)
                elif self.args.backdoor_type == 'FAST_optimize_multi_patch_in_test_free':
                    trace = self.attack_strategy.perturb(self.args, trace, y)
                elif self.args.backdoor_type == 'FAST_optimize_BackdoorRLNet_multi_patch_in':
                    trace = self.attack_strategy.perturb(self.args, trace, path=sample_path)
                elif self.args.backdoor_type == 'badnet_BackdoorRLNet_optimize_multi_patch_in' or self.args.backdoor_type == 'badnet_random_BackdoorRLNet_optimize_multi_patch_in':
                    trace = self.attack_strategy.perturb(self.args, trace, path=sample_path, label=y)
                elif self.args.backdoor_type == 'SHAP_optimize_multi_patch_in':
                    trace = self.attack_strategy.perturb(self.args, trace)
                elif self.args.backdoor_type == 'TrojanFlow':
                    trace = self.attack_strategy.perturb(self.args, trace)
                else:
                    trace = trace
                end_time = time.time()
                # print(f"trace time is: {end_time - start_time} s")
                change_in_trace = self.backdoor_length / len(original_trace)
                if self.backdoor_label_type == 'poi':
                    y = self.backdoor_lable
                elif self.backdoor_label_type == 'lc':
                    y = y
                    # y = self.backdoor_lable


        # 计算trace的变化量

        x = self.feature_transform_func(trace)
        x = torch.from_numpy(x).float()
        y = torch.tensor(y).long()

        return x, y, change_in_trace

    def __len__(self):
        return len(self.flist)


class AdpBackdoorDataset(SingleDomainDataset):
    def __init__(self, args: argparse.Namespace, flist: np.ndarray, labels: np.ndarray, is_train: bool, 
                 return_backdoored: bool = False, backdoor_lable: int = 0, backdoor_type: str = 'default'):
        super().__init__(args, flist, labels, is_train)
        self.args = args
        self.backdoor_ratio = args.backdoor_ratio
        self.return_backdoored = False if is_train else return_backdoored
        self.label_max = np.max(labels)
        self.label_min = np.min(labels)
        self.backdoor_lable = backdoor_lable
        self.backdoor_label_type = args.backdoor_label_type
        self.backdoor_length = args.backdoor_length

        # if not hasattr(self.args, 'trigger_model') and 'Transformer' in args.backdoor_type:
        #     trigger_model = InsertTransformer()
        #     # device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        #     # trigger_model = trigger_model.to(device)
        #     self.args.trigger_model = trigger_model

        # 根据 args.backdoor_type 选择攻击策略
        self.attack_strategy = self.get_attack_strategy(args.backdoor_type)

    def get_attack_strategy(self, backdoor_type: str) -> BackdoorAttackStrategy:
        strategies = {
            'default': DefaultBackdoorAttackStrategy,
            'another': AnotherBackdoorAttackStrategy,
            'badnet_random_in': BadNetRandomStrategyOnlyIn,
            'badnet_patch_in': BadNetPatchStrategyOnlyIn,
            'badnet_multi_patch_in': BadNetMultiPatchStrategyOnlyIn,
            'OSAD_optimize_multi_patch_in': OSADOptimizeMultiPatchStrategyOnlyIn,
            'DAMERAU_optimize_multi_patch_in': DAMERAUOptimizeMultiPatchStrategyOnlyIn,
            'FAST_optimize_multi_patch_in': FASTOptimizeMultiPatchStrategyOnlyIn,
            'BackdoorRLNet_optimize_multi_patch_in': BackdoorRLNetOptimizeMultiPatchStrategyOnlyIn,
            'FAST_optimize_multi_patch_in_test_free': FASTOptimizeMultiPatchStrategyOnlyInTestFree,
            'FAST_optimize_BackdoorRLNet_multi_patch_in':FASTOptimizeBackdoorRLNetMultiPatchStrategyOnlyIn,
            'badnet_BackdoorRLNet_optimize_multi_patch_in':BadNetBackdoorRLNetMultiPatchStrategyOnlyIn,
            'badnet_random_BackdoorRLNet_optimize_multi_patch_in':BadNetRandomBackdoorRLNetMultiPatchStrategyOnlyIn,
            # 可以在这里添加更多策略
        }
        return strategies.get(backdoor_type, DefaultBackdoorAttackStrategy)()

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, float):
        sample_path = self.flist[idx]
        trace = parse_trace(sample_path)
        original_trace = trace.copy()

        y = self.labels[idx]
        change_in_trace = 0.0
        if self.is_train:
            # train with backdoor_ratio, perturb the trace and change to the target label
            if self.backdoor_label_type == 'poi':
                if y != self.backdoor_lable and np.random.rand() < self.backdoor_ratio:
                    if self.args.backdoor_type == 'BackdoorRLNet_optimize_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, sample_path)
                    elif self.args.backdoor_type == 'FAST_optimize_multi_patch_in_test_free':
                        trace = self.attack_strategy.perturb(self.args, trace, train_flog=self.is_train)
                    elif self.args.backdoor_type == 'FAST_optimize_BackdoorRLNet_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, path=None, train_flog=self.is_train)
                    elif self.args.backdoor_type == 'badnet_BackdoorRLNet_optimize_multi_patch_in' or self.args.backdoor_type == 'badnet_random_BackdoorRLNet_optimize_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, path=None, train_flog=self.is_train)
                    else:
                        trace = self.attack_strategy.perturb(self.args, trace)
                    change_in_trace = self.backdoor_length / len(original_trace)
                    y = self.backdoor_lable
            elif self.backdoor_label_type == 'lc':
                if y == self.backdoor_lable and np.random.rand() < self.backdoor_ratio:
                    if self.args.backdoor_type == 'BackdoorRLNet_optimize_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, sample_path)
                    elif self.args.backdoor_type == 'FAST_optimize_multi_patch_in_test_free':
                        trace = self.attack_strategy.perturb(self.args, trace, train_flog=self.is_train)
                    elif self.args.backdoor_type == 'FAST_optimize_BackdoorRLNet_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, path=None, train_flog=self.is_train)
                    elif self.args.backdoor_type == 'badnet_BackdoorRLNet_optimize_multi_patch_in' or self.args.backdoor_type == 'badnet_random_BackdoorRLNet_optimize_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, path=None, train_flog=self.is_train)
                    else:
                        trace = self.attack_strategy.perturb(self.args, trace)
                    change_in_trace = self.backdoor_length / len(original_trace)
                    y = y

            if self.args.adp_attack == 'random_remove':
                trace = random_remove(trace, backdoor_length=self.args.backdoor_length, backdoor_num=self.args.backdoor_num)
                trace = trace[trace[:, 0].argsort()]
            elif self.args.adp_attack == 'clean_finetune':
                trace = original_trace
                y = self.labels[idx]          
        else:
            # test
            if self.return_backdoored:
                start_time = time.time()
                if self.args.backdoor_type == 'BackdoorRLNet_optimize_multi_patch_in':
                    trace = self.attack_strategy.perturb(self.args, trace, path=sample_path)
                elif self.args.backdoor_type == 'FAST_optimize_multi_patch_in_test_free':
                    trace = self.attack_strategy.perturb(self.args, trace, y)
                elif self.args.backdoor_type == 'FAST_optimize_BackdoorRLNet_multi_patch_in':
                    trace = self.attack_strategy.perturb(self.args, trace, path=sample_path)
                elif self.args.backdoor_type == 'badnet_BackdoorRLNet_optimize_multi_patch_in' or self.args.backdoor_type == 'badnet_random_BackdoorRLNet_optimize_multi_patch_in':
                    trace = self.attack_strategy.perturb(self.args, trace, path=sample_path, label=y)
                else:
                    trace = self.attack_strategy.perturb(self.args, trace)
                end_time = time.time()
                # print(f"trace time is: {end_time - start_time} s")
                change_in_trace = self.backdoor_length / len(original_trace)
                if self.backdoor_label_type == 'poi':
                    y = self.backdoor_lable
                elif self.backdoor_label_type == 'lc':
                    y = y
                    # y = self.backdoor_lable

            if self.args.adp_attack == 'random_remove':
                trace = random_remove(trace, backdoor_length=self.args.backdoor_length, backdoor_num=self.args.backdoor_num)
                trace = trace[trace[:, 0].argsort()]  

        # 计算trace的变化量

        x = self.feature_transform_func(trace)
        x = torch.from_numpy(x).float()
        y = torch.tensor(y).long()

        return x, y, change_in_trace

    def __len__(self):
        return len(self.flist)


class ClusteredDataset(Data.Dataset):
    def __init__(self, data, cluster_labels):
        self.data = data
        self.cluster_labels = cluster_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回数据和新的聚类标签
        return self.data[idx], self.cluster_labels[idx]


class VisualBackdoorDataset(SingleDomainDataset):
    def __init__(self, args: argparse.Namespace, flist: np.ndarray, labels: np.ndarray, is_train: bool, 
                 return_backdoored: bool = False, backdoor_lable: int = 0, backdoor_type: str = 'default'):
        super().__init__(args, flist, labels, is_train)
        self.args = args
        self.backdoor_ratio = args.backdoor_ratio
        self.return_backdoored = False if is_train else return_backdoored
        self.label_max = np.max(labels)
        self.label_min = np.min(labels)
        self.backdoor_lable = backdoor_lable
        self.backdoor_label_type = args.backdoor_label_type
        self.backdoor_length = args.backdoor_length

        # 根据 args.backdoor_type 选择攻击策略
        self.attack_strategy = self.get_attack_strategy(args.backdoor_type)

    def get_attack_strategy(self, backdoor_type: str) -> BackdoorAttackStrategy:
        strategies = {
            'default': DefaultBackdoorAttackStrategy,
            'another': AnotherBackdoorAttackStrategy,
            'badnet_random_in': BadNetRandomStrategyOnlyIn,
            'badnet_patch_in': BadNetPatchStrategyOnlyIn,
            'badnet_multi_patch_in': BadNetMultiPatchStrategyOnlyIn,
            'OSAD_optimize_multi_patch_in': OSADOptimizeMultiPatchStrategyOnlyIn,
            'DAMERAU_optimize_multi_patch_in': DAMERAUOptimizeMultiPatchStrategyOnlyIn,
            'FAST_optimize_multi_patch_in': FASTOptimizeMultiPatchStrategyOnlyIn,
            'BackdoorRLNet_optimize_multi_patch_in': BackdoorRLNetOptimizeMultiPatchStrategyOnlyIn,
            'FAST_optimize_multi_patch_in_test_free': FASTOptimizeMultiPatchStrategyOnlyInTestFree,
            'FAST_optimize_BackdoorRLNet_multi_patch_in':FASTOptimizeBackdoorRLNetMultiPatchStrategyOnlyIn,
            'badnet_BackdoorRLNet_optimize_multi_patch_in':BadNetBackdoorRLNetMultiPatchStrategyOnlyIn,
            'badnet_random_BackdoorRLNet_optimize_multi_patch_in':BadNetRandomBackdoorRLNetMultiPatchStrategyOnlyIn,
            # 可以在这里添加更多策略
        }
        return strategies.get(backdoor_type, DefaultBackdoorAttackStrategy)()

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, float):
        sample_path = self.flist[idx]
        trace = parse_trace(sample_path)
        original_trace = trace.copy()

        y = self.labels[idx]
        change_in_trace = 0.0
        if self.is_train:
            # train with backdoor_ratio, perturb the trace and change to the target label
            if self.backdoor_label_type == 'poi':
                if y != self.backdoor_lable and np.random.rand() < self.backdoor_ratio:
                    if self.args.backdoor_type == 'BackdoorRLNet_optimize_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, sample_path)
                    elif self.args.backdoor_type == 'FAST_optimize_multi_patch_in_test_free':
                        trace = self.attack_strategy.perturb(self.args, trace, train_flog=self.is_train)
                    elif self.args.backdoor_type == 'FAST_optimize_BackdoorRLNet_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, path=None, train_flog=self.is_train)
                    elif self.args.backdoor_type == 'badnet_BackdoorRLNet_optimize_multi_patch_in' or self.args.backdoor_type == 'badnet_random_BackdoorRLNet_optimize_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, path=None, train_flog=self.is_train)
                    else:
                        trace = self.attack_strategy.perturb(self.args, trace)
                    change_in_trace = self.backdoor_length / len(original_trace)
                    y = self.backdoor_lable
            elif self.backdoor_label_type == 'lc':
                if np.random.rand() < self.backdoor_ratio:
                    if self.args.backdoor_type == 'BackdoorRLNet_optimize_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, sample_path)
                    elif self.args.backdoor_type == 'FAST_optimize_multi_patch_in_test_free':
                        trace = self.attack_strategy.perturb(self.args, trace, train_flog=self.is_train)
                    elif self.args.backdoor_type == 'FAST_optimize_BackdoorRLNet_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, path=None, train_flog=self.is_train)
                    elif self.args.backdoor_type == 'badnet_BackdoorRLNet_optimize_multi_patch_in' or self.args.backdoor_type == 'badnet_random_BackdoorRLNet_optimize_multi_patch_in':
                        trace = self.attack_strategy.perturb(self.args, trace, path=None, train_flog=self.is_train)
                    else:
                        trace = self.attack_strategy.perturb(self.args, trace)
                    change_in_trace = self.backdoor_length / len(original_trace)
                    y = y
        else:
            # test
            if self.return_backdoored:
                start_time = time.time()
                if self.args.backdoor_type == 'BackdoorRLNet_optimize_multi_patch_in':
                    trace = self.attack_strategy.perturb(self.args, trace, path=sample_path)
                elif self.args.backdoor_type == 'FAST_optimize_multi_patch_in_test_free':
                    trace = self.attack_strategy.perturb(self.args, trace, y)
                elif self.args.backdoor_type == 'FAST_optimize_BackdoorRLNet_multi_patch_in':
                    trace = self.attack_strategy.perturb(self.args, trace, path=sample_path)
                elif self.args.backdoor_type == 'badnet_BackdoorRLNet_optimize_multi_patch_in' or self.args.backdoor_type == 'badnet_random_BackdoorRLNet_optimize_multi_patch_in':
                    trace = self.attack_strategy.perturb(self.args, trace, path=sample_path, label=y)
                else:
                    trace = self.attack_strategy.perturb(self.args, trace)
                end_time = time.time()
                # print(f"trace time is: {end_time - start_time} s")
                change_in_trace = self.backdoor_length / len(original_trace)
                if self.backdoor_label_type == 'poi':
                    y = self.backdoor_lable
                elif self.backdoor_label_type == 'lc':
                    y = y
                    # y = self.backdoor_lable


        # 计算trace的变化量

        x = self.feature_transform_func(trace)
        x = torch.from_numpy(x).float()
        y = torch.tensor(y).long()

        return x, y, change_in_trace

    def __len__(self):
        return len(self.flist)



class TrainTriggerDataset(SingleDomainDataset):
    def __init__(self, args: argparse.Namespace, flist: np.ndarray, labels: np.ndarray, is_train: bool, 
                 return_backdoored: bool = False, backdoor_lable: int = 0, backdoor_type: str = 'default'):
        super().__init__(args, flist, labels, is_train)
        self.args = args
        self.backdoor_ratio = args.backdoor_ratio
        self.return_backdoored = False if is_train else return_backdoored
        self.label_max = np.max(labels)
        self.label_min = np.min(labels)
        self.backdoor_lable = backdoor_lable
        self.backdoor_label_type = args.backdoor_label_type
        self.backdoor_length = args.backdoor_length


    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, float):
        sample_path = self.flist[idx]
        trace = parse_trace(sample_path)
        # original_trace = trace.copy()

        y = self.labels[idx]
        change_in_trace = 0.0

        x = self.feature_transform_func(trace)
        x = torch.from_numpy(trace).float()
        y = torch.tensor(y).long()

        return trace, y, change_in_trace

    def __len__(self):
        return len(self.flist)
    
    def collate_fn(batch):
        traces, labels, changes = zip(*batch)
        # traces_new = adjust_length(traces)
        
        # traces 保持原始的 numpy 数组或其他格式，不转换为 tensor
        # 直接返回原始数据
        labels = torch.stack(labels)
        changes = torch.tensor(changes)
        
        return traces, labels, changes
    

class TestTriggerDataset(SingleDomainDataset):
    def __init__(self, args: argparse.Namespace, flist: np.ndarray, labels: np.ndarray, is_train: bool, 
                 return_backdoored: bool = False, backdoor_lable: int = 0, backdoor_type: str = 'default', include_filenames=False):
        super().__init__(args, flist, labels, is_train)
        self.args = args
        self.backdoor_ratio = args.backdoor_ratio
        self.return_backdoored = False if is_train else return_backdoored
        self.label_max = np.max(labels)
        self.label_min = np.min(labels)
        self.backdoor_lable = backdoor_lable
        self.backdoor_label_type = args.backdoor_label_type
        self.backdoor_length = args.backdoor_length
        self.include_filenames = include_filenames


        # # 根据 args.backdoor_type 选择攻击策略
        # self.attack_strategy = self.get_attack_strategy(args.backdoor_type)

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, float):
        sample_path = self.flist[idx]
        trace = parse_trace(sample_path)
        # original_trace = trace.copy()

        y = self.labels[idx]

        x = self.feature_transform_func(trace)
        x = torch.from_numpy(trace).float()
        y = torch.tensor(y).long()

        if self.include_filenames:
            return trace, y, sample_path
        else:
            return trace, y

    def __len__(self):
        return len(self.flist)
    
    def collate_fn(batch):
        if len(batch[0]) == 3:
            x, y, filenames = zip(*batch)
            return list(x), list(y), list(filenames)
        else:
            x, y = zip(*batch)
            return list(x), list(y)
        

class GenerateBackdoorTraceDataset(SingleDomainDataset):
    def __init__(self, args: argparse.Namespace, flist: np.ndarray, labels: np.ndarray, is_train: bool, 
                 return_backdoored: bool = False, backdoor_lable: int = 0, backdoor_type: str = 'default', include_filenames=False):
        super().__init__(args, flist, labels, is_train)
        self.args = args
        self.backdoor_ratio = args.backdoor_ratio
        self.return_backdoored = False if is_train else return_backdoored
        self.label_max = np.max(labels)
        self.label_min = np.min(labels)
        self.backdoor_lable = backdoor_lable
        self.backdoor_label_type = args.backdoor_label_type
        self.backdoor_length = args.backdoor_length
        self.include_filenames = include_filenames
        # 根据 args.backdoor_type 选择攻击策略
        self.attack_strategy = self.get_attack_strategy(args.backdoor_type)

    def get_attack_strategy(self, backdoor_type: str) -> BackdoorAttackStrategy:
        strategies = {
            'default': DefaultBackdoorAttackStrategy,
            'another': AnotherBackdoorAttackStrategy,
            'badnet_random_in': BadNetRandomStrategyOnlyIn,
            'badnet_patch_in': BadNetPatchStrategyOnlyIn,
            'badnet_multi_patch_in': BadNetMultiPatchStrategyOnlyIn,
            'OSAD_optimize_multi_patch_in': OSADOptimizeMultiPatchStrategyOnlyIn,
            'DAMERAU_optimize_multi_patch_in': DAMERAUOptimizeMultiPatchStrategyOnlyIn,
            'FAST_optimize_multi_patch_in': FASTOptimizeMultiPatchStrategyOnlyIn,
            'BackdoorRLNet_optimize_multi_patch_in': BackdoorRLNetOptimizeMultiPatchStrategyOnlyIn
            # 可以在这里添加更多策略
        }
        return strategies.get(backdoor_type, DefaultBackdoorAttackStrategy)()

    def __getitem__(self, idx) -> (torch.Tensor, torch.Tensor, float):
        sample_path = self.flist[idx]
        trace = parse_trace(sample_path)
        original_trace = trace.copy()

        y = self.labels[idx]
        change_in_trace = 0.0
        if self.is_train:
            # train with backdoor_ratio, perturb the trace and change to the target label
            if self.backdoor_label_type == 'poi':
                if np.random.rand() < self.backdoor_ratio:
                    if self.args.backdoor_type == 'BackdoorRLNet_optimize_multi_patch_in':
                        trace_new, change_indexs = self.attack_strategy.perturb(self.args, trace, sample_path, True)
                    else:
                        trace_new, change_indexs = self.attack_strategy.perturb(self.args, trace,True)
                    y = self.backdoor_lable
            elif self.backdoor_label_type == 'lc':
                if np.random.rand() < self.backdoor_ratio:
                    if self.args.backdoor_type == 'BackdoorRLNet_optimize_multi_patch_in':
                        trace_new, change_indexs = self.attack_strategy.perturb(self.args, trace, sample_path,True)
                    else:
                        trace_new, change_indexs = self.attack_strategy.perturb(self.args, trace,True)
                    y = y
        else:
            # test
            if self.return_backdoored:
                if self.args.backdoor_type == 'BackdoorRLNet_optimize_multi_patch_in':
                    trace_new = self.attack_strategy.perturb(self.args, trace, sample_path)
                else:
                    trace_new = self.attack_strategy.perturb(self.args, trace)
                change_in_trace = self.backdoor_length / len(original_trace)
                if self.backdoor_label_type == 'poi':
                    y = self.backdoor_lable
                elif self.backdoor_label_type == 'lc':
                    y = y


        # 计算trace的变化量

        x = self.feature_transform_func(trace)
        x = torch.from_numpy(x).float()
        y = torch.tensor(y).long()

        if self.include_filenames:
            return trace_new, y, sample_path, change_indexs
        else:
            return trace, y

    def __len__(self):
        return len(self.flist)
    
    def collate_fn(batch):
        if len(batch[0]) == 4:
            x, y, filenames, change = zip(*batch)
            return list(x), list(y), list(filenames), list(change)
        else:
            x, y = zip(*batch)
            return list(x), list(y)