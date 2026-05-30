from typing import Union
from typing import Callable, Optional, Sequence, Tuple, Union
from ignite.exceptions import NotComputableError

import numpy as np
import torch
from ignite.metrics import Metric

from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

from utils.general import PR_THRES_NUM


class WFMetric(Metric):
    def __init__(self, nmc: int, device: Union[str, torch.device] = torch.device("cpu")):
        self.nmc = nmc
        self._p = 0
        self._n = 0
        self._tp = 0
        self._fp = 0
        super(WFMetric, self).__init__(device=device)

    def reset(self):
        self._p = 0
        self._n = 0
        self._tp = 0
        self._fp = 0
        super(WFMetric, self).reset()

    def update(self, output: tuple, logits: bool = True):
        """
        :param output: tuple of (y_pred, y)
        :param logits: if True, y_pred is logits, else y_pred is labels
        """
        y_pred, y = output[0].detach(), output[1].detach()
        indices = torch.argmax(y_pred, dim=1) if logits else y_pred
        idx_p = y < self.nmc
        idx_n = y == self.nmc

        self._p += torch.sum(idx_p).item()
        self._n += torch.sum(idx_n).item()

        self._tp += (indices[idx_p] == y[idx_p]).sum().item()
        self._fp += (indices[idx_n] != y[idx_n]).sum().item()

    def compute(self):
        """
        :return: tp, fp, p, n
        """
        return self._tp, self._fp, self._p, self._n

class BDWFPRCurve(Metric):
    def __init__(self, nmc: int, device: Union[str, torch.device] = torch.device("cpu"), backdoor_label: int = None):
        self.nmc = nmc  # number of monitored classes
        self.thres = np.linspace(0.01, 0.99, PR_THRES_NUM)  # threshold range
        self.backdoor_label = backdoor_label
        self._tps = np.zeros(len(self.thres))
        self._fps = np.zeros(len(self.thres))
        self._wps = np.zeros(len(self.thres))
        self._fns = np.zeros(len(self.thres))
        self._tns = np.zeros(len(self.thres))

        super(BDWFPRCurve, self).__init__(device=device)

    def reset(self):
        self._tps = np.zeros(len(self.thres))
        self._fps = np.zeros(len(self.thres))
        self._wps = np.zeros(len(self.thres))
        self._fns = np.zeros(len(self.thres))
        self._tns = np.zeros(len(self.thres))
        super(BDWFPRCurve, self).reset()

    def update(self, output: tuple, logits: bool = True):
        """
        :param output: tuple of (y_pred, y)
        :param logits: if True, y_pred is logits (before softmax)
        :param backdoor_label: label to be excluded from the calculation
        """
        y_pred, y = output[0].detach(), output[1].detach()

        if self.backdoor_label is not None:
            # Create a mask to exclude samples with the backdoor label
            mask = (y != self.backdoor_label)
            y = y[mask]
            y_pred = y_pred[mask]
            
            # Set the logits or probabilities for the backdoor label to negative infinity or zero
            # if logits:
            #     y_pred[:, self.backdoor_label] = float('-inf')
            # else:
            #     y_pred[:, self.backdoor_label] = 0.0

        if logits:
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        for i, th in enumerate(self.thres):
            confs, indices = torch.max(y_pred, dim=1)
            idx_p = y < self.nmc
            idx_n = y == self.nmc

            idx_pred_as_p = (confs >= th) & (indices < self.nmc)
            idx_pred_as_n = (confs < th) | (indices == self.nmc)

            idx_pred_eq_y = indices == y

            tp = (idx_p & idx_pred_as_p & idx_pred_eq_y).sum().item()  # monitored and predicted as monitored
            fp = (idx_n & idx_pred_as_p).sum().item()  # unmonitored but predicted as monitored
            wp = (idx_p & idx_pred_as_p & (~idx_pred_eq_y)).sum().item()  # monitored but predicted as another monitored
            fn = (idx_p & idx_pred_as_n).sum().item()  # monitored but predicted as unmonitored
            tn = (idx_n & idx_pred_as_n).sum().item()  # unmonitored and predicted as unmonitored

            self._tps[i] += tp
            self._fps[i] += fp
            self._wps[i] += wp
            self._fns[i] += fn
            self._tns[i] += tn

            assert tp + wp + fn == idx_p.sum(), "TP + WP + FN != P"
            assert fp + tn == idx_n.sum(), "FP + TN != N"
            assert idx_p.sum() + idx_n.sum() == len(y), "P + N != Total"


    def compute(self):
        """
        :return: tp, fp, wp, fn, tn
        """
        # stack to (PR_THRES_NUM, 5) shape
        stats = np.stack((self._tps, self._fps, self._wps, self._fns, self._tns), axis=1)
        return stats
    

class WFPRCurve(Metric):
    def __init__(self, nmc: int, device: Union[str, torch.device] = torch.device("cpu")):
        self.nmc = nmc  # number of monitored classes
        self.thres = np.linspace(0.01, 0.99, PR_THRES_NUM)  # threshold range

        self._tps = np.zeros(len(self.thres))
        self._fps = np.zeros(len(self.thres))
        self._wps = np.zeros(len(self.thres))
        self._fns = np.zeros(len(self.thres))
        self._tns = np.zeros(len(self.thres))

        super(WFPRCurve, self).__init__(device=device)

    def reset(self):
        self._tps = np.zeros(len(self.thres))
        self._fps = np.zeros(len(self.thres))
        self._wps = np.zeros(len(self.thres))
        self._fns = np.zeros(len(self.thres))
        self._tns = np.zeros(len(self.thres))
        super(WFPRCurve, self).reset()

    def update(self, output: tuple, logits: bool = True):
        """
        :param output: tuple of (y_pred, y)
        :param logits: if True, y_pred is logits (before softmax)
        """
        y_pred, y = output[0].detach(), output[1].detach()

        if logits:
            y_pred = torch.nn.functional.softmax(y_pred, dim=1)

        for i, th in enumerate(self.thres):
            confs, indices = torch.max(y_pred, dim=1)
            idx_p = y < self.nmc
            idx_n = y == self.nmc

            idx_pred_as_p = (confs >= th) & (indices < self.nmc)
            idx_pred_as_n = (confs < th) | (indices == self.nmc)

            idx_pred_eq_y = indices == y

            tp = (idx_p & idx_pred_as_p & idx_pred_eq_y).sum().item()  # monitored and predicted as monitored
            fp = (idx_n & idx_pred_as_p).sum().item()  # unmonitored but predicted as monitored
            wp = (idx_p & idx_pred_as_p & (~idx_pred_eq_y)).sum().item()  # monitored but predicted as another monitored
            fn = (idx_p & idx_pred_as_n).sum().item()  # monitored but predicted as unmonitored
            tn = (idx_n & idx_pred_as_n).sum().item()  # unmonitored and predicted as unmonitored

            self._tps[i] += tp
            self._fps[i] += fp
            self._wps[i] += wp
            self._fns[i] += fn
            self._tns[i] += tn

            assert tp + wp + fn == idx_p.sum(), "TP + WP + FN != P"
            assert fp + tn == idx_n.sum(), "FP + TN != N"
            assert idx_p.sum() + idx_n.sum() == len(y), "P + N != Total"

    def compute(self):
        """
        :return: tp, fp, wp, fn, tn
        """
        # stack to (PR_THRES_NUM, 5) shape
        stats = np.stack((self._tps, self._fps, self._wps, self._fns, self._tns), axis=1)
        return stats


class _BaseClassification(Metric):
    def __init__(
        self,
        output_transform: Callable = lambda x: x,
        is_multilabel: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        self._is_multilabel = is_multilabel
        self._type: Optional[str] = None
        self._num_classes: Optional[int] = None
        super(_BaseClassification, self).__init__(output_transform=output_transform, device=device)

    def reset(self) -> None:
        self._type = None
        self._num_classes = None

    def _check_shape(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output

        if not (y.ndimension() == y_pred.ndimension() or y.ndimension() + 1 == y_pred.ndimension()):
            raise ValueError(
                "y must have shape of (batch_size, ...) and y_pred must have "
                "shape of (batch_size, num_categories, ...) or (batch_size, ...), "
                f"but given {y.shape} vs {y_pred.shape}."
            )

        y_shape = y.shape
        y_pred_shape: Tuple[int, ...] = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if not (y_shape == y_pred_shape):
            raise ValueError("y and y_pred must have compatible shapes.")

        if self._is_multilabel and not (y.shape == y_pred.shape and y.ndimension() > 1 and y.shape[1] > 1):
            raise ValueError(
                "y and y_pred must have same shape of (batch_size, num_categories, ...) and num_categories > 1."
            )

    def _check_binary_multilabel_cases(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output

        if not torch.equal(y, y**2):
            raise ValueError("For binary cases, y must be comprised of 0's and 1's.")

        if not torch.equal(y_pred, y_pred**2):
            raise ValueError("For binary cases, y_pred must be comprised of 0's and 1's.")

    def _check_type(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output

        if y.ndimension() + 1 == y_pred.ndimension():
            num_classes = y_pred.shape[1]
            if num_classes == 1:
                update_type = "binary"
                self._check_binary_multilabel_cases((y_pred, y))
            else:
                update_type = "multiclass"
        elif y.ndimension() == y_pred.ndimension():
            self._check_binary_multilabel_cases((y_pred, y))

            if self._is_multilabel:
                update_type = "multilabel"
                num_classes = y_pred.shape[1]
            else:
                update_type = "binary"
                num_classes = 1
        else:
            raise RuntimeError(
                f"Invalid shapes of y (shape={y.shape}) and y_pred (shape={y_pred.shape}), check documentation."
                " for expected shapes of y and y_pred."
            )
        if self._type is None:
            self._type = update_type
            self._num_classes = num_classes
        else:
            if self._type != update_type:
                raise RuntimeError(f"Input data type has changed from {self._type} to {update_type}.")
            if self._num_classes != num_classes:
                raise ValueError(f"Input data number of classes has changed from {self._num_classes} to {num_classes}")


class ASR(_BaseClassification):
    _state_dict_all_req_keys = ("_num_correct", "_num_examples")

    def __init__(
        self,
        backdoor_lable: int = 0,
        output_transform: Callable = lambda x: x,
        is_multilabel: bool = False,
        device: Union[str, torch.device] = torch.device("cpu"),
    ):
        self.backdoor_lable = backdoor_lable
        super(ASR, self).__init__(output_transform=output_transform, is_multilabel=is_multilabel, device=device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._num_correct = torch.tensor(0, device=self._device)
        self._num_examples = 0
        super(ASR, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._check_shape(output)
        self._check_type(output)
        y_pred, y = output[0].detach(), output[1].detach()
        y = torch.full_like(y, self.backdoor_lable)

        if self._type == "binary":
            correct = torch.eq(y_pred.view(-1).to(y), y.view(-1))
        elif self._type == "multiclass":
            indices = torch.argmax(y_pred, dim=1)
            correct = torch.eq(indices, y).view(-1)
        elif self._type == "multilabel":
            # if y, y_pred shape is (N, C, ...) -> (N x ..., C)
            num_classes = y_pred.size(1)
            last_dim = y_pred.ndimension()
            y_pred = torch.transpose(y_pred, 1, last_dim - 1).reshape(-1, num_classes)
            y = torch.transpose(y, 1, last_dim - 1).reshape(-1, num_classes)
            correct = torch.all(y == y_pred.type_as(y), dim=-1)

        self._num_correct += torch.sum(correct).to(self._device)
        self._num_examples += correct.shape[0]

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self) -> float:
        if self._num_examples == 0:
            raise NotComputableError("Accuracy must have at least one example before it can be computed.")
        return self._num_correct.item() / self._num_examples
