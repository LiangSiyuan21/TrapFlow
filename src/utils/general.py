import os
import random
from functools import wraps
from pathlib import Path
from time import strftime
from time import time
from typing import Tuple, Union, Callable, Optional, Any, List

import numpy as np
import pandas as pd
import torch

PR_THRES_NUM = 10  # number of thresholds for precision-recall curve


def seed_everything(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def increment_path(path: os.PathLike, exist_ok: bool = False, sep: str = '', mkdir: bool = False) -> Path:
    """
    Create the file or directory path without incrementing. If the path exists,
    the existing path is returned if `exist_ok` is True. Optionally create the directory if `mkdir` is True.
    """
    path = Path(path)  # os-agnostic
    
    if path.exists():
        return path
    
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path

def timeit(f: Callable):
    @wraps(f)
    def wrap(*args: Optional[Any], **kw: Optional[Any]) -> float:
        ts = time()
        result = f(*args, **kw)
        te = time()
        print('func:%r took: %2.4f sec' % (f.__name__, te - ts))
        return result

    return wrap


def parse_trace(fdir: str, sanity_check: bool = False) -> np.ndarray:
    """
    Parse a trace file based on our predefined format
    :param sanity_check: whether to perform sanity check on the trace

    Pay attention: do not enable sanity check on TrafficSliver dataset
    """
    trace = pd.read_csv(fdir, delimiter="\t", header=None)
    trace = np.array(trace)
    
    if sanity_check:
        # it is possible the trace has a long tail
        # if there is a time gap between two bursts larger than CUT_OFF_THRESHOULD
        # We cut off the trace here sicne it could be a long timeout or
        # maybe the loading is already finished
        # Set a very conservative value
        CUT_OFF_THRESHOLD = 15
        start, end = 0, len(trace)
        ipt_burst = np.diff(trace[:, 0])
        ipt_outlier_inds = np.where(ipt_burst > CUT_OFF_THRESHOLD)[0]

        if len(ipt_outlier_inds) > 0:
            outlier_ind_first = ipt_outlier_inds[0]
            if outlier_ind_first < 50:
                start = outlier_ind_first + 1
            outlier_ind_last = ipt_outlier_inds[-1]
            if outlier_ind_last > 50:
                end = outlier_ind_last + 1
        trace = trace[start:end].copy()

        # remove the first few lines that are incoming packets
        start = -1
        for _, size in trace:
            start += 1
            if size > 0:
                break

        trace = trace[start:].copy()
        trace[:, 0] -= trace[0, 0]
        assert trace[0, 0] == 0
    return trace


def parse_trace_trafficsliver(fdir: str) -> List[np.ndarray]:
    # Read the file into a DataFrame, treating blank lines as NaN
    df = pd.read_csv(fdir, delimiter="\t", header=None, skip_blank_lines=False)

    # Identify indices of blank lines
    blank_line_indices = df[df.isna().all(axis=1)].index

    # Split the DataFrame into multiple DataFrames based on blank lines
    arrays = []
    previous_index = 0

    for index in blank_line_indices:
        if previous_index != index:  # Ensure we don't append empty arrays
            arr = np.array(df.iloc[previous_index:index].dropna().reset_index(drop=True))
            arrays.append(arr)
        previous_index = index + 1

    # Append the last section if it's not empty
    if previous_index < len(df):
        arrays.append(np.array(df.iloc[previous_index:].dropna().reset_index(drop=True)))

    return arrays


def tam(sample: np.ndarray, time_window: float, max_load_time: float, pad_length: int = None) -> np.ndarray:
    """Extract the tam feature from a trace."""
    cut_off_time = min(max_load_time, float(sample[-1, 0]))
    num_bins = int(cut_off_time / time_window) + 1
    bins = np.linspace(0, num_bins * time_window, num_bins).tolist() + [np.inf]

    outgoing = sample[np.sign(sample[:, 1]) > 0]
    incoming = sample[np.sign(sample[:, 1]) < 0]

    cnt_outgoing, _ = np.histogram(outgoing[:, 0], bins=bins)
    cnt_incoming, _ = np.histogram(incoming[:, 0], bins=bins)
    # merge to 2d feature
    feat = np.stack((cnt_outgoing, cnt_incoming), axis=1)
    assert feat.flatten().sum() == len(sample), \
        "Sum of feature ({}) is not equal to the length of the trace ({}). BUG?".format(
            feat.flatten().sum(), len(sample))

    if pad_length is not None:
        if len(feat) < pad_length:
            pad = np.zeros((pad_length - len(feat), feat.shape[1]))
            feat = np.concatenate((feat, pad))
        else:
            feat = feat[:pad_length, :]
    return feat


def get_ipt(sample: np.ndarray, time_window: float, max_load_time: float, granularity: int,
            pad_length: int = None) -> np.ndarray:
    """Extract the ipt feature from a trace."""
    cut_off_time = min(max_load_time, float(sample[-1, 0]))
    num_bins = int(cut_off_time / time_window) + 1
    bins = np.linspace(0, num_bins * time_window, num_bins).tolist() + [np.inf]

    mask_outgoing = sample[:, 1] > 0
    idx_bin = np.digitize(sample[:, 0], bins)

    ipts = np.diff(sample[:, 0], prepend=0)

    ipt_bins = np.logspace(-6, -1, granularity).tolist() + [np.inf]
    ipt_bins[0] = 0

    feat_outgoing, _, _ = np.histogram2d(sample[:, 0][mask_outgoing], ipts[mask_outgoing], bins=(bins, ipt_bins))
    feat_incoming, _, _ = np.histogram2d(sample[:, 0][~mask_outgoing], ipts[~mask_outgoing], bins=(bins, ipt_bins))

    # zip two features
    feat = np.stack((feat_outgoing, feat_incoming), axis=-1)
    feat = feat.reshape(-1, granularity * 2)  # [outgoing, incoming, outgoing, incoming, ...]

    # feat = np.concatenate((feat_outgoing, feat_incoming), axis=1)

    assert feat.flatten().sum() == len(sample), \
        "Sum of feature ({}) is not equal to the length of the trace ({}). BUG?".format(
            feat.flatten().sum(), len(sample))

    if pad_length is not None:
        if len(feat) < pad_length:
            pad = np.zeros((pad_length - len(feat), feat.shape[1]))
            feat = np.concatenate((feat, pad))
        else:
            feat = feat[:pad_length, :]
    return feat


def feature_transform(sample: np.ndarray, feature_type: str, seq_length: int, n_tam: int = 1, granularity: int = 4) \
        -> np.ndarray:
    """
    Transform a raw sample to the specific feature space.
    :return a numpy array of shape (1 or 2, seq_length)
    """
    if feature_type == 'df':
        feat = np.sign(sample[:, 1])

    elif feature_type == 'tiktok':
        feat = sample[:, 0] * np.sign(sample[:, 1])

    elif feature_type == 'tam':
        max_load_time = 80  # s
        time_window = 0.044  # s
        feat = tam(sample, time_window, max_load_time, pad_length=None)

    elif feature_type == 'tam+':
        max_load_time = 80
        time_window = 0.044  # s
        pad_length = None

        feat = []
        for i in range(n_tam):
            time_window = time_window * (1 + 0.5 * i)
            feat_once = tam(sample, time_window, max_load_time, pad_length=pad_length)
            pad_length = len(feat_once)
            feat.append(feat_once)
        feat = np.concatenate(feat, axis=1)

    elif feature_type == 'fusion':
        max_load_time = 80
        time_window = 0.044  # s

        # ipt
        feat1 = get_ipt(sample, time_window, max_load_time, granularity=granularity, pad_length=None)

        # # tam
        # feat2 = tam(sample, time_window, max_load_time, pad_length=None)
        #
        # feat = np.concatenate((feat1, feat2), axis=1)
        feat = feat1

    elif feature_type == 'patch':
        patch_size = 20
        dim = 5

        direction = np.sign(sample[:, 1])
        time = sample[:, 0]
        ipt = np.diff(time, prepend=0)

        # pad to a multiple of patch_size
        if len(sample) % patch_size != 0:
            pad = np.zeros((patch_size - len(sample) % patch_size))
            direction = np.concatenate((direction, pad))
            time = np.concatenate((time, pad))
            ipt = np.concatenate((ipt, pad))

        N = len(ipt) // patch_size

        is_padding = np.array([0] * len(sample) + [1] * (len(ipt) - len(sample))).astype(bool)

        is_outgoing = direction > 0

        time_bins = [0] + np.logspace(-5, -1, dim - 1).tolist() + [np.inf]

        feat_mask = np.zeros((len(time_bins) - 1, len(ipt))).astype(bool)  # B x L

        for i in range(len(time_bins) - 1):
            mask = (ipt >= time_bins[i]) & (ipt < time_bins[i + 1])
            feat_mask[i] = mask

        feat_mask = feat_mask.reshape(dim, N, patch_size)  # B x L/P x P

        is_outgoing = is_outgoing.reshape(N, patch_size)  # L/P x P
        is_padding = is_padding.reshape(N, patch_size)  # L/P x P

        feat = np.zeros((len(ipt) // patch_size, dim * 2))
        for i in range(0, len(feat_mask)):
            feat[:, 2 * i] = (feat_mask[i] * is_outgoing * (~is_padding)).sum(axis=1)  # outgoing
            feat[:, 2 * i + 1] = (feat_mask[i] * ~is_outgoing * (~is_padding)).sum(axis=1)  # incoming

        assert feat.sum() == len(sample), \
            "Sum of burst lengths ({}) is not equal to the length of the trace ({}). BUG?".format(sum(feat),
                                                                                                  len(sample))
        # # packets per second
        # time = time.reshape(N, patch_size)
        # time_gap = time.max(axis=1) - time[:, 0] + 1e-7  # consider there is padding, we have to use max
        #
        # feat_packet_per_second_out = feat[:, ::2].sum(axis=1) / time_gap  # outgoing
        # feat_packet_per_second_in = feat[:, 1::2].sum(axis=1) / time_gap  # incoming
        #
        # feat = np.concatenate((feat, feat_packet_per_second_out[:, np.newaxis], feat_packet_per_second_in[:, np.newaxis]), axis=1)

    elif feature_type == 'burst':
        sample = sample[:, 1]
        # Create a mask for consecutive elements that are the same
        mask = np.where(np.sign(sample[:-1]) != np.sign(sample[1:]))[0] + 1
        mask = np.concatenate((mask, [len(sample)]))  # add the last index
        # Count the number of elements between sign changes
        feat = np.diff(mask, prepend=0)
        assert sum(feat) == len(sample), \
            "Sum of burst lengths ({}) is not equal to the length of the trace ({}). BUG?".format(sum(feat),
                                                                                                  len(sample))
    else:
        raise NotImplementedError("Feature type {} is not implemented.".format(feature_type))

    # make sure 2d
    if len(feat.shape) == 1:
        feat = feat[:, np.newaxis]
    # pad to seq_length
    if len(feat) < seq_length:
        pad = np.zeros((seq_length - len(feat), feat.shape[1]))
        feat = np.concatenate((feat, pad))
    feat = feat[:seq_length, :]
    return np.transpose(feat, (1, 0))


def feature_transform_from_path(fdir: str, feature_type: str, seq_length: int, n_tam: int = 1,
                                sanity_check: bool = True) -> np.ndarray:
    """
    Transform a raw sample to the specific feature space.
    :return a numpy array of shape (1 or 2, seq_length)
    """
    sample = parse_trace(fdir, sanity_check)
    return feature_transform(sample, feature_type, seq_length, n_tam)


def get_flist_label_single_domain(data_path: Union[str, os.PathLike], mon_cls: int, mon_inst: int, unmon_inst: int,
                                  suffix: str = '.cell') \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a list of file paths and corresponding labels.
    :param data_path: the path to the data directory
    :param mon_cls: number of monitored classes
    :param mon_inst: number of monitored instances per class
    :param unmon_inst: number of unmonitored instances
    :param suffix: file suffix
    :return: a list of file paths and a list of corresponding labels
    """
    flist = []
    labels = []
    for cls in range(mon_cls):
        for inst in range(mon_inst):
            pth = os.path.join(data_path, '{}-{}{}'.format(cls, inst, suffix))
            if os.path.exists(pth):
                flist.append(pth)
                labels.append(cls)
    for inst in range(unmon_inst):
        pth = os.path.join(data_path, '{}{}'.format(inst, suffix))
        if os.path.exists(pth):
            flist.append(pth)
            labels.append(mon_cls)

    assert len(flist) > 0, "No files found in {}!".format(data_path)
    return np.array(flist), np.array(labels)


def get_flist_label_multi_domain(data_path: Union[str, os.PathLike], mon_cls: int, mon_inst: int, unmon_inst: int,
                                 page_per_class: int = 1, suffix: str = '.cell') \
        -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Generate a list of file paths and corresponding labels.
    :param data_path: the path to the data directory
    :param mon_cls: number of monitored classes
    :param mon_inst: number of monitored instances per class
    :param unmon_inst: number of unmonitored instances
    :param page_per_class: number of pages per class
    :param suffix: file suffix
    :return: a list of file paths and a list of corresponding labels, num of monitored site classes
    """
    flist = []
    labels = []
    max_mon_label = -1
    for cls in range(mon_cls):
        for inst in range(mon_inst):
            pth = os.path.join(data_path, '{}-{}{}'.format(cls, inst, suffix))
            if os.path.exists(pth):
                flist.append(pth)
                labels.append(cls // page_per_class)
                if cls // page_per_class > max_mon_label:
                    max_mon_label = cls // page_per_class

    for inst in range(unmon_inst):
        pth = os.path.join(data_path, '{}{}'.format(inst, suffix))
        if os.path.exists(pth):
            flist.append(pth)
            labels.append(max_mon_label + 1)

    assert len(flist) > 0, "No files found in {}!".format(data_path)
    return max_mon_label + 1, np.array(flist), np.array(labels)


def return_loading_bandwidth(fdir: Union[str, os.PathLike]) -> float:
    """Return the loading bandwidth of a trace."""
    trace = parse_trace(fdir)
    return len(trace) / (float(trace[-1, 0]) + 1e-5)


def select_fast_slow(flist: Union[List, np.ndarray], labels: Union[List, np.ndarray],
                     train_mode: str = 'fast', ratio: float = 0.9, n_jobs: int = 20) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    pick the top/bottom ratio samples as train
    return train&test
    """
    import multiprocessing as mp

    train_list, test_list = [], []
    train_labels, test_labels = [], []

    for label in np.unique(labels):
        _flist = flist[labels == label]
        with mp.Pool(n_jobs) as p:
            loading_bandwidths = p.map(return_loading_bandwidth, _flist)
        loading_bandwidths = np.array(loading_bandwidths)
        if train_mode == 'fast':
            indices = np.argsort(-loading_bandwidths)
        else:
            indices = np.argsort(loading_bandwidths)
        n_train = int(len(indices) * ratio)
        train_list.extend(_flist[indices[:n_train]])
        train_labels.extend([label] * n_train)

        test_list.extend(_flist[indices[n_train:]])
        test_labels.extend([label] * (len(indices) - n_train))

    return np.array(train_list), np.array(train_labels), np.array(test_list), np.array(test_labels)


def init_directories(output_parent_dir: Union[str, os.PathLike], defense_name: str) -> str:
    # Create a results dir if it doesn't exist yet
    if not os.path.exists(output_parent_dir):
        os.makedirs(output_parent_dir)

    # Define output directory
    timestamp = strftime('%m%d_%H%M%S')
    output_dir = os.path.join(output_parent_dir, defense_name + '_' + timestamp)
    os.makedirs(output_dir)
    return output_dir


def mean_by_label(samples: torch.tensor, labels: torch.tensor) -> (torch.tensor, torch.tensor):
    """ select mean(samples), count() from samples group by labels order by labels asc """
    weight = torch.zeros(int(labels.max()) + 1, samples.shape[0]).to(samples.device)  # L, N
    weight[labels, torch.arange(samples.shape[0])] = 1
    label_count = weight.sum(dim=1)
    weight = torch.nn.functional.normalize(weight, p=1, dim=1)  # l1 normalization
    mean = torch.mm(weight, samples)  # L, F
    index = torch.arange(mean.shape[0])[label_count > 0]
    return mean[index], label_count[index]


# Function to calculate gradient norms
def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm
