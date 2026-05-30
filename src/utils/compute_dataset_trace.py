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
    return flist


data_path = '/mnt/hdd/liujiayang/liangsiyuan/WFTransfer/sirinam_filtered/'
mon_classes = 100
mon_inst = 100
unmon_inst = 0
page_per_class = 1
suffix = '.cell'

flist  = get_flist_label_multi_domain(data_path, mon_classes, mon_inst, unmon_inst, page_per_class, suffix)
print(len(flist))
