import argparse
import os
from pathlib import Path
from typing import Tuple
import json
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

from attacks import Attack
# from attacks.modules import DFNet, InceptionNet, TMWF_DFNet, RFNet, RFNet2, RFNet3, RFNet4
from attacks.modules import DFNet, InceptionNet, RFNet, RFNet2, RFNet3, RFNet4, TMWF, ARES, VarCNN
from attacks.modules.kfingerprinting import KFPConfig, KFingerprintingRF
from utils.data import BackdoorDataset
from utils.general import get_flist_label_multi_domain, increment_path, PR_THRES_NUM, get_grad_norm, select_fast_slow
from utils.metric import WFMetric, WFPRCurve, ASR, BDWFPRCurve



def generate_log_name(trigger_pth, default_log_name="training.log"):
    # 提取文件路径中的目录名称
    trigger_dir_name = os.path.dirname(trigger_pth).split('/')[-1]
    
    # 提取目录名称中的前三个由下划线分隔的部分
    split_dir_name = trigger_dir_name.split('_')
    first_three_parts = '_'.join(split_dir_name[:3])  # 只取前3个部分
    
    # 提取 4000_4 部分
    patch_info = '_'.join([split_dir_name[-2], split_dir_name[-1]])  # 提取倒数第三和倒数第二部分
    
    # 提取 epoch 信息，并移除文件名的后缀
    trigger_file_name = os.path.basename(trigger_pth).replace('.json', '')  # 移除 .json 后缀
    split_file_name = trigger_file_name.split('_')
    epoch_info = next((s for s in split_file_name if 'epoch' in s), '')

    # 组合 log 名称，保留默认日志名的前缀和提取的关键信息
    log_name_prefix = default_log_name.split('.')[0]
    log_name = f"{log_name_prefix}_{first_three_parts}_{patch_info}_{epoch_info}.log"
    
    return log_name

def create_model_save_path(log_name):
    # 去掉log文件的后缀，只保留用于文件夹名称的部分
    dir_name = log_name.replace(".log", "")
    
    return dir_name

class FinetuneTest(Attack):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)

        last_part = self.args.data_path.rstrip('/').split('/')[-1]
        self.dataset_name = last_part.split('_')[0]

        # log path
        if self.args.open_world:
            log_path = increment_path(
                Path(self.args.model_path) / "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                    self.dataset_name, self.args.model, self.args.feature_type,
                    self.args.backdoor_type, str(self.args.backdoor_length),
                    str(self.args.backdoor_ratio), str(self.args.backdoor_label_type),
                    str(self.args.backdoor_lable), 'open'
                ),
                sep='_', exist_ok=self.args.exist_ok, mkdir=True
            )
        else:
            log_path = increment_path(
                Path(self.args.model_path) / "{}_{}_{}_{}_{}_{}_{}_{}".format(
                    self.dataset_name, self.args.model, self.args.feature_type,
                    self.args.backdoor_type, str(self.args.backdoor_length),
                    str(self.args.backdoor_ratio), str(self.args.backdoor_label_type),
                    str(self.args.backdoor_lable)
                ),
                sep='_', exist_ok=self.args.exist_ok, mkdir=True
            )

        log_dir = Path(log_path) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        self.log_name = "training.log"
        if self.args.trigger_pth is not None:
            self.log_name = generate_log_name(self.args.trigger_pth, default_log_name=self.log_name)
        log_file = log_dir / self.log_name

        # reset handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        self.logger.info("Logger configured to write to file: {}".format(log_file))

        assert self.args.aug_times >= 0
        assert self.args.averaging_times > 0
        assert self.args.page_per_class > 0

        self.args.unmon_inst = args.unmon_inst if args.open_world else 0

        self.nmc, self.flist, self.labels = get_flist_label_multi_domain(
            self.args.data_path,
            mon_cls=self.args.mon_classes,
            mon_inst=self.args.mon_inst,
            unmon_inst=self.args.unmon_inst,
            page_per_class=self.args.page_per_class,
            suffix=self.args.suffix
        )

        self.nc = len(np.unique(self.labels))
        self.unmon_inst = args.unmon_inst if args.open_world else 0

        assert self.nc == self.nmc + int(self.args.open_world)

        self.logger.info("Number of data: {}, Num of classes: {} + {}".format(
            len(self.flist), self.nmc, self.nc - self.nmc
        ))

        self.feature_type = self.args.feature_type
        self.aug_times = args.aug_times

        self.amp_mode = 'amp' if (not self.args.not_amp) and self.device != torch.device("cpu") else None

        self.logger.info('Augmentation times: {} | Averaging times: {}'.format(
            self.args.aug_times, self.args.averaging_times
        ))

        # checkpoint path
        if not self.args.nosave and self.args.mode == 'train':
            if self.args.open_world:
                save_pth = Path(self.args.model_path) / "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
                    self.dataset_name, self.args.model, self.args.feature_type,
                    self.args.backdoor_type, str(self.args.backdoor_length),
                    str(self.args.backdoor_ratio), str(self.args.backdoor_label_type),
                    str(self.args.backdoor_lable), 'open'
                )
            else:
                save_pth = Path(self.args.model_path) / "{}_{}_{}_{}_{}_{}_{}_{}".format(
                    self.dataset_name, self.args.model, self.args.feature_type,
                    self.args.backdoor_type, str(self.args.backdoor_length),
                    str(self.args.backdoor_ratio), str(self.args.backdoor_label_type),
                    str(self.args.backdoor_lable)
                )
            if self.args.trigger_pth is not None:
                model_save_path = create_model_save_path(self.log_name)
                save_pth = save_pth / model_save_path

            self.checkpoint_path = increment_path(
                save_pth, sep='_', exist_ok=self.args.exist_ok, mkdir=True
            )

    def _build_model(self):
        if self.args.verbose:
            self.logger.info("Building model: {} | Feature: {}".format(self.args.model, self.args.feature_type))

        ch = 2 if self.feature_type == 'tam' or self.feature_type == 'tam+' else 1

        if self.args.model == 'df':
            model = DFNet(length=self.args.seq_length, num_classes=self.nc, in_channels=ch)

        elif self.args.model == 'inception':
            if self.feature_type == 'tam' or self.feature_type == 'tam+':
                model = InceptionNet(length=self.args.seq_length, num_classes=self.nc, in_channels=1,
                                     num_kernels=self.args.num_kernels)
            else:
                model = InceptionNet(length=self.args.seq_length, num_classes=self.nc,
                                     in_channels=self.args.fusion_granularity,
                                     num_kernels=self.args.num_kernels)

        elif self.args.model == 'tmwf':
            assert self.feature_type == 'df' or self.feature_type == 'tiktok', \
                "TMWF only supports DF or TikTok features"
            assert self.args.seq_length == 30720
            model = TMWF(num_classes=self.nc)

        elif self.args.model == 'varcnn':
            assert self.feature_type == 'tiktok', \
                "VarCNN only supports DF or TikTok features"
            model = VarCNN(num_classes=self.nc)

        elif self.args.model == 'ares':
            assert self.feature_type == 'df' or self.feature_type == 'tiktok', \
                "ARES only supports DF or TikTok features"
            model = ARES(num_classes=self.nc)

        elif self.args.model == 'rf':
            assert self.feature_type == 'tam' or self.feature_type == 'tam+' or self.feature_type == 'fusion', \
                "RF only supports TAM or fusion features"
            if self.feature_type == 'tam' or self.feature_type == 'tam+':
                model = RFNet(num_classes=self.nc)
            else:
                model = RFNet(num_classes=self.nc, in_channel=self.args.fusion_granularity + 1)

        elif self.args.model == 'rf2':
            assert self.feature_type == 'tam' or self.feature_type == 'tam+' or self.feature_type == 'fusion', \
                "RF2 only supports TAM or fusion features"
            if self.feature_type == 'tam' or self.feature_type == 'tam+':
                model = RFNet2(num_classes=self.nc, in_channel=1)
            else:
                model = RFNet2(num_classes=self.nc, in_channel=self.args.fusion_granularity)

        elif self.args.model == 'rf3':
            assert self.feature_type == 'tam' or self.feature_type == 'tam+' or self.feature_type == 'fusion', \
                "RF3 only supports TAM or fusion features"
            if self.feature_type == 'tam' or self.feature_type == 'tam+':
                model = RFNet3(num_classes=self.nc, in_channel=1)
            else:
                model = RFNet3(num_classes=self.nc, in_channel=self.args.fusion_granularity)

        elif self.args.model == 'rf4':
            assert self.feature_type == 'tam' or self.feature_type == 'tam+' or self.feature_type == 'fusion', \
                "RF4 only supports TAM or fusion features"
            if self.feature_type == 'tam' or self.feature_type == 'tam+':
                model = RFNet4(num_classes=self.nc, in_channel=1)
            else:
                model = RFNet4(num_classes=self.nc, in_channel=self.args.fusion_granularity,
                               num_kernels_1d=self.args.num_kernels, num_kernels_2d=self.args.num_kernels)

        elif self.args.model == 'kfingerprinting':
            # 这里不 build torch 模型：在 train() 里单独处理 fit+eval
            raise RuntimeError("kfingerprinting is non-differentiable; handled in train() branch.")

        else:
            raise NotImplementedError("Model {} is not implemented.".format(self.args.model))

        return model.to(self.device)

    def _get_data(self, flist: np.ndarray, labels: np.ndarray, is_train: bool = True,
                  return_backdoored: bool = False, backdoor_lable: int = 0, backdoor_type: str = 'default') -> Tuple[BackdoorDataset, DataLoader]:
        batch_size = self.args.batch_size
        dataset = BackdoorDataset(
            self.args, flist, labels, is_train,
            return_backdoored=return_backdoored,
            backdoor_lable=backdoor_lable,
            backdoor_type=backdoor_type
        )
        print(f"Using {backdoor_type} backdoor attack strategy. label attack is {self.args.backdoor_label_type}.")
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=self.args.workers)
        return dataset, loader

    def _loader_to_numpy_xy(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for batch in loader:
            x, y, _ = batch
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            Xs.append(x.reshape(x.shape[0], -1))  # flatten
            ys.append(y)
        X = np.concatenate(Xs, axis=0).astype(np.float32)
        Y = np.concatenate(ys, axis=0).astype(np.int64)
        return X, Y

    def run(self, one_fold_only: bool = False):
        # trigger preload (原样保留)
        if self.args.backdoor_type == 'FAST_optimize_multi_patch_in_test_free':
            self.args.test_trigger = None
            if self.args.trigger_pth is not None:
                with open(self.args.trigger_pth, 'r') as f:
                    data = json.load(f)
                self.args.test_trigger = {key: value for key, value in data.items() if ".cell" not in key}
        elif self.args.backdoor_type in [
            'FAST_optimize_BackdoorRLNet_multi_patch_in',
            'badnet_BackdoorRLNet_optimize_multi_patch_in',
            'badnet_random_BackdoorRLNet_optimize_multi_patch_in',
            "BackdoorRLNet_optimize_multi_patch_in"
        ]:
            self.args.test_trigger = None
            if self.args.trigger_pth is not None:
                with open(self.args.trigger_pth, 'r') as f:
                    data = json.load(f)
                self.args.test_trigger = data

        if self.args.open_world:
            res = np.zeros((PR_THRES_NUM, 5))
        else:
            res = np.zeros(4)

        sss = StratifiedShuffleSplit(n_splits=10, test_size=0.1, random_state=self.args.seed)

        for fold, (train_index, test_index) in enumerate(sss.split(self.flist, self.labels)):
            if one_fold_only and fold > 0:
                break

            _train_list, _train_labels = self.flist[train_index], self.labels[train_index]

            train_list, val_list, train_labels, val_labels = train_test_split(
                _train_list, _train_labels,
                test_size=0.10,
                random_state=self.args.seed,
                stratify=_train_labels
            )

            if len(val_list) == len(val_labels) and len(val_list) > self.args.eval_nums:
                val_list = val_list[:self.args.eval_nums]
                val_labels = val_labels[:self.args.eval_nums]

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

            res_one_fold = self.train(
                fold + 1,
                train_list_new, train_labels_new,
                val_list, val_labels,
                test_list, test_labels,
                self.args.backdoor_lable, self.args.backdoor_type
            )
            res += res_one_fold
            self.logger.info("-" * 10)

        if self.args.open_world:
            precisions = res[:, 0] / (res[:, 0] + res[:, 1] + res[:, 2] + 1e-6)
            recalls = res[:, 0] / (res[:, 0] + res[:, 2] + res[:, 3] + 1e-6)
            for i in range(PR_THRES_NUM):
                print("{:.4f} {:.4f}".format(precisions[i], recalls[i]))
        else:
            print("{:.0f} {:.0f} {:.0f} {:.0f}".format(res[0], res[1], res[2], res[3]))

    @staticmethod
    def train_step(engine: Engine, batch: Tuple, model: nn.Module, optimizer: torch.optim, criterion: nn.Module,
                   device: torch.device, scaler: torch.cuda.amp.GradScaler, clip_value: float, use_amp: bool):
        model.train()
        optimizer.zero_grad()
        x, y, _ = batch
        x, y = x.to(device), y.to(device)

        if use_amp:
            with torch.cuda.amp.autocast():
                y_pred = model(x)
                loss = criterion(y_pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            scaler.step(optimizer)
            scaler.update()
        else:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

        return loss.item()

    def create_supervised_trainer(self, model: nn.Module, optimizer: torch.optim, criterion: nn.Module,
                                  device: torch.device = None, clip_value: float = 1.0,
                                  use_amp: bool = False) -> Engine:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        return Engine(lambda engine, batch: self.train_step(
            engine, batch, model, optimizer, criterion, device, scaler, clip_value, use_amp
        ))

    def train(self, fold: int, train_list: np.ndarray, train_labels: np.ndarray,
              val_list: np.ndarray, val_labels: np.ndarray,
              test_list: np.ndarray, test_labels: np.ndarray,
              backdoor_lable: int = 0, backdoor_type: str = 'default') -> np.ndarray:

        # dataloaders
        _, train_loader = self._get_data(train_list, train_labels, is_train=True,
                                         return_backdoored=False, backdoor_lable=backdoor_lable, backdoor_type=backdoor_type)
        _, val_loader = self._get_data(val_list, val_labels, is_train=False,
                                       return_backdoored=False, backdoor_lable=backdoor_lable, backdoor_type=backdoor_type)
        _, test_loader = self._get_data(test_list, test_labels, is_train=False,
                                        return_backdoored=False, backdoor_lable=backdoor_lable, backdoor_type=backdoor_type)

        backdoored_val_loader = None
        backdoored_test_loader = None
        if self.args.asr:
            _, backdoored_val_loader = self._get_data(val_list, val_labels, is_train=False, return_backdoored=True,
                                                      backdoor_lable=backdoor_lable, backdoor_type=backdoor_type)
            _, backdoored_test_loader = self._get_data(test_list, test_labels, is_train=False, return_backdoored=True,
                                                       backdoor_lable=backdoor_lable, backdoor_type=backdoor_type)

        # =========================
        # k-fingerprinting branch
        # =========================
        if self.args.model == 'kfingerprinting':
            X_train, y_train = self._loader_to_numpy_xy(train_loader)

            cfg = KFPConfig(
                n_estimators=getattr(self.args, "n_estimators", 500),
                k=getattr(self.args, "k", 3),
                unanimous=getattr(self.args, "unanimous", True),
                random_state=getattr(self.args, "seed", 0),
            )
            kfp = KFingerprintingRF(cfg).fit(X_train, y_train)

            class _KFPWrapper(nn.Module):
                def __init__(self, kfp_model: KFingerprintingRF, device: torch.device, open_world: bool, unmon_label: int):
                    super().__init__()
                    self.kfp = kfp_model
                    self.device = device
                    self.open_world = open_world
                    self.unmon_label = unmon_label

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    x_np = x.detach().cpu().numpy().reshape(x.size(0), -1).astype(np.float32)
                    proba = self.kfp.predict_proba(x_np)

                    if self.open_world:
                        row_sum = proba.sum(axis=1)
                        zero_rows = (row_sum == 0)
                        if np.any(zero_rows):
                            proba[zero_rows, self.unmon_label] = 1.0

                    eps = 1e-6
                    logits = np.log(np.clip(proba, eps, 1.0))
                    return torch.from_numpy(logits).to(self.device)

            torch_model = _KFPWrapper(
                kfp_model=kfp,
                device=self.device,
                open_world=self.args.open_world,
                unmon_label=self.nmc if self.args.open_world else 0
            ).to(self.device)

            # NLLLoss because forward returns log-prob
            criterion = nn.NLLLoss()
            res = self.test(torch_model, test_loader, backdoored_test_loader, criterion, backdoor_lable)

            # kFP 没有 state_dict 可 save；你要存的话得 joblib.dump（需要我再给你加）
            return res

        # =========================
        # normal torch training
        # =========================
        model = self._build_model()
        criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)

        if self.args.mode == 'test':
            model.load_state_dict(torch.load(self.args.model_path, map_location='cpu'))
            res = self.test(model, test_loader, backdoored_test_loader, criterion, backdoor_lable)
        else:
            if self.args.pretrained:
                if self.args.verbose:
                    self.logger.info(f"Loading model from {self.args.pretrained}")
                model = self.load_from_checkpoint(model, self.args.pretrained)

            lr0 = self.args.lr0
            self.logger.info(f"Initial learning rate: {lr0}")
            optimizer = torch.optim.Adam(model.parameters(), lr=lr0, weight_decay=self.args.weight_decay)

            step_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.2 ** (epoch / self.args.epochs))
            lr_scheduler = LRScheduler(step_scheduler)

            trainer = self.create_supervised_trainer(
                model, optimizer, criterion, self.device, clip_value=5, use_amp=not self.args.not_amp
            )

            backdoor_val_metrics = {"asr": ASR(backdoor_lable=backdoor_lable)}

            val_metrics = {
                "accuracy": WFMetric(self.nmc),
                "acc": Accuracy(),
                "loss": Loss(criterion)
            }
            if self.args.asr and self.args.open_world:
                val_metrics = {
                    "accuracy": BDWFPRCurve(self.nmc, backdoor_label=self.args.backdoor_lable),
                    "acc": Accuracy(),
                    "loss": Loss(criterion)
                }

            val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=self.device, amp_mode=self.amp_mode)
            test_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=self.device, amp_mode=self.amp_mode)

            if self.args.asr:
                if self.args.open_world:
                    backdoor_val_metrics = {"pr": BDWFPRCurve(self.nmc, backdoor_label=self.args.backdoor_lable)}
                    backdoor_val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=self.device, amp_mode=self.amp_mode)
                    backdoor_test_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=self.device, amp_mode=self.amp_mode)
                else:
                    if self.args.adversarial_state is True:
                        backdoor_val_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=self.device, amp_mode=self.amp_mode)
                        backdoor_test_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=self.device, amp_mode=self.amp_mode)
                    else:
                        backdoor_val_evaluator = create_supervised_evaluator(model, metrics=backdoor_val_metrics, device=self.device, amp_mode=self.amp_mode)
                        backdoor_test_evaluator = create_supervised_evaluator(model, metrics=backdoor_val_metrics, device=self.device, amp_mode=self.amp_mode)

            @trainer.on(Events.EPOCH_COMPLETED)
            def log_training_loss(engine: Engine):
                if self.args.verbose:
                    grad_norm = get_grad_norm(model)
                    self.logger.info(
                        f"Fold[{fold}] | Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] | "
                        f"Loss: {engine.state.output:.2f} | Norm: {grad_norm:.4f}"
                    )

            @trainer.on(Events.EPOCH_COMPLETED)
            def print_lr():
                if self.args.verbose:
                    self.logger.info(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

            @trainer.on(Events.EPOCH_COMPLETED)
            def log_validation_results(engine: Engine):
                val_evaluator.run(val_loader)
                test_evaluator.run(test_loader)
                clean_val_metrics = val_evaluator.state.metrics
                clean_test_metrics = test_evaluator.state.metrics
                backdoor_val_metrics_local = None
                backdoor_test_metrics_local = None

                if self.args.asr:
                    backdoor_val_evaluator.run(backdoored_val_loader)
                    backdoor_test_evaluator.run(backdoored_test_loader)
                    backdoor_val_metrics_local = backdoor_val_evaluator.state.metrics
                    backdoor_test_metrics_local = backdoor_test_evaluator.state.metrics

                if self.args.verbose:
                    if self.args.open_world:
                        _metrics = clean_val_metrics
                        res_pr = _metrics['accuracy']
                        precisions = res_pr[:, 0] / (res_pr[:, 0] + res_pr[:, 1] + res_pr[:, 2] + 1e-6)
                        recalls = res_pr[:, 0] / (res_pr[:, 0] + res_pr[:, 2] + res_pr[:, 3] + 1e-6)

                        self.logger.info(
                            f"Validation Results - Fold[{fold}] Epoch[{engine.state.epoch}] | "
                            f"Avg loss: {_metrics['loss']:.2f} | Acc: {_metrics['acc']:.2f}"
                        )
                        for i in range(PR_THRES_NUM):
                            self.logger.info(f"Clean Threshold {i}: Precision: {precisions[i]:.4f}, Recall: {recalls[i]:.4f}")

                        _bd = backdoor_val_metrics_local
                        res_bd = _bd['pr']
                        precisions_bd = res_bd[:, 0] / (res_bd[:, 0] + res_bd[:, 1] + res_bd[:, 2] + 1e-6)
                        recalls_bd = res_bd[:, 0] / (res_bd[:, 0] + res_bd[:, 2] + res_bd[:, 3] + 1e-6)
                        for i in range(PR_THRES_NUM):
                            self.logger.info(f"Backdoor Threshold {i}: Precision: {precisions_bd[i]:.4f}, Recall: {recalls_bd[i]:.4f}")

                        _metrics = clean_test_metrics
                        res_pr = _metrics['accuracy']
                        precisions = res_pr[:, 0] / (res_pr[:, 0] + res_pr[:, 1] + res_pr[:, 2] + 1e-6)
                        recalls = res_pr[:, 0] / (res_pr[:, 0] + res_pr[:, 2] + res_pr[:, 3] + 1e-6)

                        self.logger.info(
                            f"Test Results - Fold[{fold}] Epoch[{engine.state.epoch}] | "
                            f"Avg loss: {_metrics['loss']:.2f} | Acc: {_metrics['acc']:.2f}"
                        )
                        for i in range(PR_THRES_NUM):
                            self.logger.info(f"Clean Threshold {i}: Precision: {precisions[i]:.4f}, Recall: {recalls[i]:.4f}")

                        _bd = backdoor_test_metrics_local
                        res_bd = _bd['pr']
                        precisions_bd = res_bd[:, 0] / (res_bd[:, 0] + res_bd[:, 1] + res_bd[:, 2] + 1e-6)
                        recalls_bd = res_bd[:, 0] / (res_bd[:, 0] + res_bd[:, 2] + res_bd[:, 3] + 1e-6)
                        for i in range(PR_THRES_NUM):
                            self.logger.info(f"Backdoor Threshold {i}: Precision: {precisions_bd[i]:.4f}, Recall: {recalls_bd[i]:.4f}")

                    else:
                        _metrics = clean_val_metrics
                        if self.args.adversarial_state is True:
                            asr_info = f" | ACC: {backdoor_val_metrics_local['acc']:.2f}" if self.args.asr else ""
                        else:
                            asr_info = f" | ASR: {backdoor_val_metrics_local['asr']:.2f}" if self.args.asr else ""

                        self.logger.info(
                            f"Validation Results - Fold[{fold}] Epoch[{engine.state.epoch}] | "
                            f"Avg loss: {_metrics['loss']:.2f} | "
                            f"tp: {_metrics['accuracy'][0]:4.0f} fp: {_metrics['accuracy'][1]:4.0f} "
                            f"p: {_metrics['accuracy'][2]:4.0f} n: {_metrics['accuracy'][3]:4.0f} "
                            f"| Acc: {_metrics['acc']:.2f}{asr_info}"
                        )

                        if self.args.adversarial_state is True:
                            asr_info = f" | ACC: {backdoor_test_metrics_local['acc']:.2f}" if self.args.asr else ""
                        else:
                            asr_info = f" | ASR: {backdoor_test_metrics_local['asr']:.2f}" if self.args.asr else ""

                        _metrics = clean_test_metrics
                        self.logger.info(
                            f"Test Results - Fold[{fold}] Epoch[{engine.state.epoch}] | "
                            f"Avg loss: {_metrics['loss']:.2f} | "
                            f"tp: {_metrics['accuracy'][0]:4.0f} fp: {_metrics['accuracy'][1]:4.0f} "
                            f"p: {_metrics['accuracy'][2]:4.0f} n: {_metrics['accuracy'][3]:4.0f} "
                            f"| Acc: {_metrics['acc']:.2f}{asr_info}"
                        )

            trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)
            trainer.run(train_loader, max_epochs=self.args.epochs)

            res = self.test(model, test_loader, backdoored_test_loader, criterion, backdoor_lable)

        # save model (only torch models)
        if not self.args.nosave and self.args.mode == 'train':
            name = 'pretrained_finetuned_fold{}.pth'.format(fold) if self.args.pretrained else 'finetuned_{}.pth'.format(fold)
            torch.save(model.state_dict(), self.checkpoint_path / name)
            if self.args.verbose:
                self.logger.info(f"Model saved at {self.checkpoint_path}")

        torch.cuda.empty_cache()
        return res

    def test(self, model: nn.Module, test_loader: DataLoader, backdoor_test_loader: DataLoader,
             criterion: nn.Module, backdoor_lable: int = 0) -> np.ndarray:

        backdoor_val_metrics = {"asr": ASR(backdoor_lable=backdoor_lable)}
        val_metrics = {
            "accuracy": WFMetric(self.nmc),
            "acc": Accuracy(),
            "loss": Loss(criterion),
            "pr": WFPRCurve(self.nmc)
        }

        test_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=self.device, amp_mode=self.amp_mode)
        if backdoor_test_loader:
            backdoor_test_evaluator = create_supervised_evaluator(model, metrics=backdoor_val_metrics, device=self.device, amp_mode=self.amp_mode)

        test_evaluator.run(test_loader)
        metrics = test_evaluator.state.metrics

        backdoor_metrics = None
        if backdoor_test_loader is not None:
            backdoor_test_evaluator.run(backdoor_test_loader)
            backdoor_metrics = backdoor_test_evaluator.state.metrics

        if self.args.verbose:
            asr_info = f" | ASR: {backdoor_metrics['asr']:.2f}" if backdoor_test_loader is not None else ""
            self.logger.info(
                f"Test Results - Avg loss: {metrics['loss']:.2f} | "
                f"tp: {metrics['accuracy'][0]:4.0f} fp: {metrics['accuracy'][1]:4.0f} "
                f"p: {metrics['accuracy'][2]:4.0f} n: {metrics['accuracy'][3]:4.0f} "
                f"| Acc: {metrics['acc']:.2f}{asr_info}"
            )

        if self.args.open_world:
            return metrics['pr']
        else:
            return np.array(metrics['accuracy'])

    @staticmethod
    def load_from_checkpoint(model: nn.Module, path: os.PathLike, freeze: bool = False) -> nn.Module:
        checkpoint = torch.load(path)
        if 'model' in checkpoint:
            pretrained_dict = checkpoint['model']
        else:
            pretrained_dict = checkpoint
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'fc' not in k}
        model.load_state_dict(pretrained_dict, strict=False)

        for param in model.parameters():
            param.requires_grad = True

        if freeze:
            for name, param in model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False
        return model