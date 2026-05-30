import argparse
import random
from datetime import datetime
from functools import partial
from typing import Callable, Literal

import numpy as np

from utils.general import parse_trace

DUMMY = 888
PARAMETER_MAX = 0.5
PARAMETER_MIN = 0.


class TransformFunction(object):
    """Wraps the Transform function for pretty printing options."""

    def __init__(self, func: Callable, name: str):
        self.f = func
        self.name = name

    def __repr__(self):
        return '<' + self.name + '>'

    def __call__(self, trace: np.ndarray) -> np.ndarray:
        return self.f(trace)


class TransformT(object):
    """Each instance of this class represents a specific transform."""

    def __init__(self, name: str, xform_fn: Callable):
        self.name = name
        self.xform = xform_fn

    def __repr__(self) -> str:
        return '<' + self.name + '>'

    def tr_transformer(self, probability: float, level: float, feature_transform_fn: Callable = None) \
            -> TransformFunction:
        def return_function(tr: np.ndarray) -> np.ndarray:
            if random.random() < probability:
                tr = self.xform(tr, level)
            return tr

        def return_function_with_feature(tr: np.ndarray) -> np.ndarray:
            if random.random() < probability:
                tr = self.xform(tr, level, feature_transform_fn)
            return tr

        name = self.name + '({:.1f},{})'.format(probability, level)

        if feature_transform_fn is None:
            return TransformFunction(return_function, name)
        else:
            return TransformFunction(return_function_with_feature, name)


identity = TransformT('identity', lambda trace, level: trace)


def _uniformDrop_imp(trace: np.ndarray, level: float, sign: Literal[1, -1]) -> np.ndarray:
    trace_dummy = trace[trace[:, 1] == sign * DUMMY]
    trace_real = trace[trace[:, 1] != sign * DUMMY]
    num_dummy = len(trace_dummy)

    num = int(num_dummy * level)
    if num == 0:
        return trace

    idx = np.random.choice(len(trace_dummy), num, replace=False)
    out_trace_dummy = np.delete(trace_dummy, idx, axis=0)
    out_trace = np.concatenate((trace_real, out_trace_dummy), axis=0)
    out_trace = out_trace[out_trace[:, 0].argsort(kind='mergesort')]
    return out_trace


uniformDropOut = TransformT('uniformDrop', partial(_uniformDrop_imp, sign=1))
uniformDropIn = TransformT('uniformDrop', partial(_uniformDrop_imp, sign=-1))


def _uniformAdd_imp(trace: np.ndarray, level: float, sign: Literal[1, -1]) -> np.ndarray:
    num_dummy = (trace[:, 1] == sign * DUMMY).sum()
    add_num = int(num_dummy * level)
    if add_num == 0:
        return trace
    ts = np.random.uniform(trace[0, 0], trace[-1, 0], add_num)
    trace_dummy = np.stack((ts, sign * DUMMY * np.ones(len(ts))), axis=1)
    out_trace = np.concatenate((trace, trace_dummy), axis=0)
    out_trace = out_trace[out_trace[:, 0].argsort(kind='mergesort')]
    return out_trace


uniformAddOut = TransformT('uniformAdd', partial(_uniformAdd_imp, sign=1))
uniformAddIn = TransformT('uniformAdd', partial(_uniformAdd_imp, sign=-1))

UNIFORM_TRANSFORMS = [
    uniformDropOut,
    uniformDropIn,
    uniformAddOut,
    uniformAddIn,
]


def _frontAdd_impl(trace: np.ndarray, level: float, w_min: float, w_max: float, sign: Literal[1, -1]) -> np.ndarray:
    num_dummy = (trace[:, 1] == sign * DUMMY).sum()
    last_ts = trace[-1, 0]

    wnd = np.random.uniform(w_min, w_max)
    num = int(num_dummy * level)
    if num == 0:
        return trace
    ts = np.sort(np.random.rayleigh(wnd, num))
    ts = ts[ts <= last_ts]
    new_dummy = np.stack((ts, sign * DUMMY * np.ones(len(ts))), axis=1)
    trace = np.concatenate((trace, new_dummy), axis=0)
    trace = trace[trace[:, 0].argsort(kind='mergesort')]
    return trace


frontAddOut = TransformT('frontAddOut', partial(_frontAdd_impl, w_min=1., w_max=14., sign=1))
frontAddIn = TransformT('frontAddIn', partial(_frontAdd_impl, w_min=1., w_max=14., sign=-1))


def _frontDel_impl(trace: np.ndarray, level: float, w_min: float, w_max: float, sign: Literal[1, -1]) -> np.ndarray:
    trace_dummy = trace[trace[:, 1] == sign * DUMMY]
    trace_real = trace[trace[:, 1] != sign * DUMMY]
    num_dummy = len(trace_dummy)
    last_ts = trace[-1, 0]

    wnd = np.random.uniform(w_min, w_max)
    num = int(num_dummy * level)
    if num == 0:
        return trace
    ts = np.sort(np.random.rayleigh(wnd, num))

    # delete those dummy packets that are close to ts
    ts = ts[ts <= last_ts]
    idx = np.abs(trace_dummy[:, 0] - ts[:, np.newaxis]).argmin(axis=1)
    out_trace_dummy = np.delete(trace_dummy, idx, axis=0)
    out_trace = np.concatenate((trace_real, out_trace_dummy), axis=0)
    out_trace = out_trace[out_trace[:, 0].argsort(kind='mergesort')]
    return out_trace


frontDelOut = TransformT('frontDelOut', partial(_frontDel_impl, w_min=1., w_max=14., sign=1))
frontDelIn = TransformT('frontDelIn', partial(_frontDel_impl, w_min=1., w_max=14., sign=-1))

FRONT_TRANSFORMS = [
    frontAddOut,
    frontAddIn,
    frontDelOut,
    frontDelIn,
]

averaging = TransformT('averaging', lambda trace, level: trace)


def _gridMask_impl(trace: np.ndarray, level: float, feature_transform_func: Callable, padding: str) -> np.ndarray:
    trace_out = feature_transform_func(trace)  # C x L
    trace_sum = trace_out.sum(axis=0)
    # find the last non-negative position
    length = np.where(trace_sum > 0)[0][-1] + 1
    masked_length = int(length * level)
    if masked_length == 0:
        return trace_out

    mask = np.random.choice(length, masked_length, replace=False)

    if padding == 'zero':
        trace_out[:, mask] = 0
    elif padding == 'mean':
        trace_mean = np.round(trace_out.mean(axis=1, keepdims=True)).astype(int)  # C x 1
        # each channel fill with its mean value
        trace_out[:, mask] = trace_mean
    elif padding == 'max':
        trace_max = trace_out.max(axis=1, keepdims=True)
        trace_out[:, mask] = trace_max
    elif padding == 'neighbor':
        cols = mask
        left_cols = np.where(cols > 0, cols - 1, cols)
        right_cols = np.where(cols < length - 1, cols + 1, cols)
        trace_out[:, cols] = (trace_out[:, left_cols] + trace_out[:, right_cols]) / 2.
        trace_out = np.round(trace_out).astype(int)
    else:
        raise ValueError(f"padding {padding} is not supported.")
    return trace_out


def _gridSwap_impl(trace: np.ndarray, level: float, feature_transform_func: Callable) -> np.ndarray:
    trace_out = feature_transform_func(trace)  # C x L
    trace_sum = trace_out.sum(axis=0)
    # find the last non-negative position
    length = np.where(trace_sum > 0)[0][-1] + 1

    # find the number of columns to swap
    num_swap = int(length * level)
    if num_swap == 0:
        return trace_out

    # find the columns to swap
    for i in range(trace_out.shape[0]):
        cols = np.random.choice(length, num_swap, replace=False)
        cols = np.sort(cols)

        # find the columns to swap with
        cols_swap = cols + np.random.choice([-1, 1], num_swap)
        cols_swap = np.clip(cols_swap, 0, length - 1)

        # swap the columns
        trace_out[i, cols], trace_out[i, cols_swap] = trace_out[i, cols_swap], trace_out[i, cols]
    return trace_out


def _gridMerge_impl(trace: np.ndarray, level: float, feature_transform_func: Callable) -> np.ndarray:
    trace_out = feature_transform_func(trace)  # C x L
    trace_sum = trace_out.sum(axis=0)
    # find the last non-negative position
    length = np.where(trace_sum > 0)[0][-1] + 1

    # find the number of columns to merge
    num_merge = int(length * level)
    if num_merge == 0:
        return trace_out

    for i in range(trace_out.shape[0]):
        # find the columns to merge
        cols = np.random.choice(length, num_merge, replace=False)
        cols = np.sort(cols)

        # find the columns to merge with
        cols_merge = cols + np.random.choice([-1, 1], num_merge)
        cols_merge = np.clip(cols_merge, 0, length - 1)

        # merge the columns
        trace_out[i, cols] = trace_out[i, cols] + trace_out[i, cols_merge]
        trace_out[i, cols_merge] = 0

    return trace_out


def _gridAdd_impl(trace: np.ndarray, level: float, feature_transform_func: Callable) -> np.ndarray:
    trace_out = feature_transform_func(trace)  # C x L
    trace_sum = trace_out.sum(axis=0)
    # find the last non-negative position
    length = np.where(trace_sum > 0)[0][-1] + 1

    # find the number of columns to add
    num_add = int(length * level)
    if num_add == 0:
        return trace_out

    for i in range(trace_out.shape[0]):
        # find the columns to add
        cols = np.random.choice(length, num_add, replace=False)
        cols = np.sort(cols)
        # add the columns
        trace_out[i, cols] = trace_out[i, cols] + 1

    return np.round(trace_out)


def _gridDrop_impl(trace: np.ndarray, level: float, feature_transform_func: Callable) -> np.ndarray:
    trace_out = feature_transform_func(trace)  # C x L
    trace_sum = trace_out.sum(axis=0)
    # find the last non-negative position
    length = np.where(trace_sum > 0)[0][-1] + 1

    # find the number of columns to drop
    num_drop = int(length * level)
    if num_drop == 0:
        return trace_out

    for i in range(trace_out.shape[0]):
        # find the columns to drop
        cols = np.random.choice(length, num_drop, replace=False)
        cols = np.sort(cols)
        # drop the columns
        trace_out[i, cols] = trace_out[i, cols] - 1

    trace_out = np.clip(trace_out, 0, None)

    return np.round(trace_out)


# gridMaskZero = TransformT('gridMask', partial(_gridMask_impl, padding='zero'))
# gridMaskMean = TransformT('gridMask', partial(_gridMask_impl, padding='mean'))
# gridMaskMax = TransformT('gridMask', partial(_gridMask_impl, padding='max'))
# gridMaskNeighbor = TransformT('gridMask', partial(_gridMask_impl, padding='neighbor'))
gridSwap = TransformT('gridSwap', _gridSwap_impl)
gridMerge = TransformT('gridMerge', _gridMerge_impl)
gridAdd = TransformT('gridAdd', _gridAdd_impl)
gridDrop = TransformT('gridDrop', _gridDrop_impl)

GRID_MASK_TRANSFORMS = [
    # gridMaskZero,
    # gridMaskMean,
    # gridMaskMax,
    # gridMaskNeighbor,
    gridSwap,
    gridMerge,
    # gridAdd,
    # gridDrop,
]

# ALL_TRANSFORMS = UNIFORM_TRANSFORMS \
#                  + FRONT_TRANSFORMS \
#                  + [averaging]

TAM_TRANSFORMS = GRID_MASK_TRANSFORMS + UNIFORM_TRANSFORMS + FRONT_TRANSFORMS + [averaging]

PACKET_TRANSFORMS = UNIFORM_TRANSFORMS + FRONT_TRANSFORMS + [averaging]


# ALL_TRANSFORMS = [averaging]


class TrivialAugment(object):
    def __init__(self, args: argparse.Namespace, flist: np.ndarray, labels: np.ndarray,
                 feature_transform_func: Callable,
                 averaging_times: int = 4, min_level: float = PARAMETER_MIN, max_level: float = PARAMETER_MAX):
        self.args = args
        self.flist = flist
        self.labels = labels
        self.averaging_times = averaging_times
        self.feature_transform_func = feature_transform_func
        self.max_level = max_level
        self.min_level = min_level

    def __call__(self, idx: int, trace: np.ndarray) -> np.ndarray:
        # pay attention that numpy may have the same random seed for a batch of multiprocessing processes
        # https://github.com/numpy/numpy/issues/9650
        # https://stackoverflow.com/questions/67691168/how-to-generate-different-random-values-at-each-subprocess-during-a-multiprocess
        np.random.seed(datetime.now().microsecond)
        random.seed(datetime.now().microsecond)

        if self.args.feature_type == 'tiktok' or self.args.feature_type == 'df':
            ALL_TRANSFORMS = PACKET_TRANSFORMS
        else:
            ALL_TRANSFORMS = TAM_TRANSFORMS
        op = random.choices(ALL_TRANSFORMS, k=1)[0]
        if op in GRID_MASK_TRANSFORMS:
            level = random.uniform(self.min_level, self.max_level)
            res = op.tr_transformer(1., level, feature_transform_fn=self.feature_transform_func)(trace)
            return res
        elif op == averaging:
            # pick averaging_times samples from the same class
            trace = self.feature_transform_func(trace)
            similar_traces = np.random.choice(self.flist[self.labels == self.labels[idx]], self.averaging_times)
            for similar_trace in similar_traces:
                similar_trace = parse_trace(similar_trace)
                similar_trace = self.feature_transform_func(similar_trace)
                trace = trace + similar_trace
            trace = trace / (self.averaging_times + 1)
            trace = np.round(trace)
        else:
            level = random.uniform(0, self.max_level)
            trace = op.tr_transformer(1., level)(trace)
            return self.feature_transform_func(trace)
        # print(level, op.name)
        return trace
