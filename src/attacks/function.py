import time
import numpy as np
import torch
import torch.nn as nn
from ignite.contrib.handlers.param_scheduler import LRScheduler

feature_transform_func = None

import torch
import torch.nn as nn

# def sample_action(position_logits, count_logits, total_inserts=4000, num_positions=4):
#     # 计算插入位置的概率分布
#     position_probs = torch.softmax(position_logits, dim=1)  # [batch_size, seq_len]
#     # 对数概率
#     position_log_probs = torch.log(position_probs + 1e-8)
#     # insert_counts = F.softplus(count_logits)
#     # insert_counts = torch.round(insert_counts)
#     # 采样插入位置
#     positions = []
#     position_log_probs_sampled = []
#     for probs, log_probs in zip(position_probs, position_log_probs):
#         _, sampled_positions = torch.topk(probs, num_positions)
#         # sampled_positions = torch.multinomial(probs, num_positions, replacement=False)  # [num_positions]
#         positions.append(sampled_positions)
#         position_log_probs_sampled.append(log_probs[sampled_positions])
#     positions = torch.stack(positions)  # [batch_size, num_positions]
#     position_log_probs_sampled = torch.stack(position_log_probs_sampled)  # [batch_size, num_positions]
    
#     # 对应的插入数量 logits
#     counts_logits_selected = []
#     for i in range(count_logits.size(0)):
#         counts_logits_selected.append(count_logits[i][positions[i]])
#     counts_logits_selected = torch.stack(counts_logits_selected)  # [batch_size, num_positions]
    
#     # 计算插入数量的概率分布
#     counts_probs = torch.softmax(counts_logits_selected, dim=1)  # [batch_size, num_positions]
#     counts_log_probs = torch.log(counts_probs + 1e-8)
    
#     # 根据概率分配插入数量，确保总插入数量为 total_inserts
#     counts = counts_probs * total_inserts  # [batch_size, num_positions]
    
#     return positions, counts, position_log_probs_sampled, counts_log_probs

def sample_action(probs, num_positions=4):
    # probs: [batch_size, seq_len]
    topk_probs, topk_indices = probs.topk(num_positions, dim=1)
    return topk_indices  # [batch_size, k]


def insert_backdoor(trace_tensor, positions, total_inserts=4000, num_positions=4):
    """
    在指定位置插入指定数量的 -1
    """
    batch_size = trace_tensor.size(0)
    trace_bd_list = []
    
    for i in range(batch_size):
        seq = trace_tensor[i]
        pos = positions[i]
        cnt = int(total_inserts/num_positions)
        cnt = [cnt] * num_positions

        
        seq_bd = seq.clone()
        inserts = []
    
        epsilon = 1e-1
        for p, c in sorted(zip(pos.tolist(), cnt.tolist()), reverse=True):
            p = int(p)  # 确保p是整数
            c = int(c)  # 确保c是整数

            # 获取p位置的时间和前一个位置的时间
            time_p = seq_bd[p, 0]
            if p > 0:
                time_prev = seq_bd[p-1, 0] 
            else: 
                seq_bd[p, 0]  # 如果p是0，则使用相同时间

            if time_p == time_prev:
                time_p += epsilon

            # 生成c个时间值，介于time_prev和time_p之间
            if c > 1:  # 确保当c为1时不会出错
                # random_times = torch.linspace(time_prev, time_p, steps=c, device=time_prev.device)
                random_times = torch.full((c,), fill_value=time_prev.item(), device=time_prev.device, dtype=time_prev.dtype)
            else:
                random_times = torch.tensor([time_prev], device=time_prev.device)  # 只有一个时间点时

            # 构建插入值，第一列是时间，第二列是-1
            if c > 0:
                insert_value = torch.stack([random_times, -torch.ones(c).to(seq.device)], dim=1).to(seq.device)
                # 将新值插入到seq_bd中
                seq_bd = torch.cat([seq_bd[:p], insert_value, seq_bd[p:]], dim=0)

        seq_bd = seq_bd[seq_bd[:, 0].argsort()]
        trace_bd_list.append(seq_bd)
    
    return trace_bd_list

def cosine_distance_per_sample(x, y):
    """
    Compute the cosine distance between corresponding samples in two batches.

    Args:
        x (torch.Tensor): Tensor of shape [batch_size, seq_length, num_features]
        y (torch.Tensor): Tensor of shape [batch_size, seq_length, num_features]

    Returns:
        torch.Tensor: Tensor of cosine distances of shape [batch_size]
    """
    # Normalize each sample to unit vector
    x_norm = x / torch.norm(x, p=2, dim=(1, 2), keepdim=True)
    y_norm = y / torch.norm(y, p=2, dim=(1, 2), keepdim=True)
    
    # Compute the cosine similarity for each pair of samples
    cosine_similarities = (x_norm * y_norm).sum(dim=(1, 2))
    
    # Convert cosine similarity to cosine distance
    cosine_distances = 1 - cosine_similarities
    
    return cosine_distances

import torch

def batch_hamming_distance_with_padding(x, y):
    """
    计算两个批次中对应样本的汉明距离，忽略 x 中填充值 0 的位置。

    参数：
        x (torch.Tensor): 形状为 [batch_size, seq_length, 2]，x[:, :, 1] 包含值 +1、-1 或 0。
        y (torch.Tensor): 形状为 [batch_size, seq_length, 2]。

    返回：
        torch.Tensor: 形状为 [batch_size]，包含每个样本对的汉明距离。
    """
    # 获取 x 和 y 中的值部分
    x_values = x[:, :, 1]  # 形状：[batch_size, seq_length]
    y_values = y[:, :, 1]  # 形状：[batch_size, seq_length]

    # 创建一个掩码，标记 x 中值不为 0 的位置
    mask = x_values != 0  # 形状：[batch_size, seq_length]，数据类型为 bool

    # 在有效位置上比较 x_values 和 y_values
    differences = (x_values != y_values).float() * mask.float()  # 形状：[batch_size, seq_length]

    # 对序列长度维度求和，得到每个样本的汉明距离
    hamming_distances = differences.sum(dim=1)  # 形状：[batch_size]

    return hamming_distances


def batch_fast_levenshtein_like_distance(ori_traces, new_traces, cost_id=1, cost_trans=0.01):
    batch_size = ori_traces.size(0)
    device = ori_traces.device
    total_costs = torch.zeros(batch_size, dtype=torch.float32, device=device)
    
    for b in range(batch_size):
        ori_trace = ori_traces[b]
        new_trace = new_traces[b]
        m = ori_trace.size(0)
        n = new_trace.size(0)

        D = {}
        for i in range(m):
            value = ori_trace[i, 1].item()
            if value not in D:
                D[value] = []
            D[value].append(i + 1)  # 1-based index

        total_cost = 0
        for j in range(n):
            value = new_trace[j, 1].item()
            if value in D and D[value]:
                i1 = D[value].pop(0)
                transposition_cost = abs(i1 - (j + 1)) * cost_trans
                total_cost += transposition_cost
            else:
                total_cost += cost_id

        total_cost += sum(len(v) for v in D.values()) * cost_id
        total_costs[b] = total_cost

    return total_costs

# GPU版本的fast_levenshtein_like_distance函数
def fast_levenshtein_like_distance(ori_trace, new_trace, cost_id=1, cost_trans=0.01):
    device = ori_trace.device  # 获取输入张量的设备（CPU或GPU）
    m = ori_trace.size(0)
    n = new_trace.size(0)

    # 创建 ori_trace 中元素的字典 D，存储位置
    D = {}
    for i in range(m):
        value = ori_trace[i, 1].item()  # 获取 trace 的值
        if value not in D:
            D[value] = []
        D[value].append(i + 1)  # 1-based index

    total_cost = 0  # 初始化总成本

    # 遍历 new_trace 计算转置代价或插入/删除代价
    for j in range(n):
        value = new_trace[j, 1].item()
        if value in D and D[value]:
            i1 = D[value].pop(0)  # 获取并删除首次出现的位置
            transposition_cost = abs(i1 - (j + 1)) * cost_trans
            total_cost += transposition_cost
        else:
            total_cost += cost_id  # 插入或删除代价

    # 计算 D 中剩余的元素（未匹配的）的代价
    total_cost += len(D) * cost_id

    return torch.tensor(total_cost, dtype=torch.float32, device=device)

def adjust_length(trace, max_length=10000):
    current_length = trace.shape[0]
    if current_length > max_length:
        # 如果长度超过10000，截断
        return trace[:max_length]
    elif current_length < max_length:
        # 如果长度不足10000，用特定的方式填充
        padding_length = max_length - current_length
        # 获取最后一个时间戳，并生成连续的时间戳
        last_timestamp = trace[-1, 0]
        additional_timestamps = last_timestamp + np.arange(1, padding_length + 1) * 0.111111  # 根据给定步长生成时间戳

        # 创建填充数组，时间戳在第一列，第二列填充为0
        padding = np.zeros((padding_length, 2))
        padding[:, 0] = additional_timestamps
        combined = np.vstack((trace, padding))
        sorted_trace = combined[combined[:, 0].argsort()]

        return sorted_trace
    else:
        return trace
    
def compute_reward(trace_tensor, trace_bd_list, total_inserts_pred, total_inserts_target, distance='levenshtein',beta=0):
    # 将列表转换为张量
    trace_bd_tensor = torch.stack(trace_bd_list)  # 假设 trace_bd_list 已经是一个张量列表

    # 计算序列差异
    rewards = torch.empty(trace_tensor.size(0), device=trace_tensor.device)

    # loss
    s = time.time()
    # for i, (orig_seq, bd_seq) in enumerate(zip(trace_tensor, trace_bd_tensor)):
    #     # rewards[i] = F.mse_loss(orig_seq, bd_seq, reduction='mean')
    #     rewards[i] = fast_levenshtein_like_distance(orig_seq, bd_seq)
    if distance == 'levenshtein':
        rewards = batch_fast_levenshtein_like_distance(trace_tensor, trace_bd_tensor)
    elif distance == 'hamming':
        rewards = batch_hamming_distance_with_padding(trace_tensor, trace_bd_tensor)

    # rewards = cosine_distance_per_sample(trace_tensor, trace_bd_tensor)
    # rewards = F.mse_loss(trace_tensor, trace_bd_tensor, reduction='none')
    # rewards = rewards.mean(dim=[1, 2])

    e = time.time()
    # print(e-s)

    # 插入数量惩罚
    # insert_penalty = (total_inserts_pred - total_inserts_target) ** 2
    # insert_penalty = insert_penalty.clone().detach().float().to(trace_tensor.device)


    # 总奖励
    # total_reward = rewards - beta * insert_penalty  # beta 为权重系数，需要调整
    total_reward = rewards

    return total_reward


# 损失函数类
class TITCriterion(nn.Module):
    def __init__(self, cost_id=1, cost_trans=0.01, epsilon=1e-6):
        super(TITCriterion, self).__init__()
        self.cost_id = cost_id
        self.cost_trans = cost_trans
        self.epsilon = epsilon  # 防止除零

    def forward(self, ori_trace, new_trace):
        # 计算 fast_levenshtein_like_distance 的距离
        distance = fast_levenshtein_like_distance(ori_trace, new_trace, 
                                                  cost_id=self.cost_id, 
                                                  cost_trans=self.cost_trans)
        # 损失 = 1 / (距离 + epsilon)，以防 distance 为 0
        loss = 1 / (distance + self.epsilon)
        return loss

class TITCriterion_v(nn.Module):
    def __init__(self, cost_id=1, cost_trans=0.01, epsilon=1e-6):
        super(TITCriterion_v, self).__init__()
        self.cost_id = cost_id
        self.cost_trans = cost_trans
        self.epsilon = epsilon  # 防止除零

    # def forward(self, ori_trace, new_trace):
    #     # 计算两个 trace 在值上的差异
    #     diff = torch.abs(ori_trace[:, 1] - new_trace[:, 1])
        
    #     # 计算转置成本
    #     transposition_cost = torch.abs(torch.arange(len(ori_trace), device=ori_trace.device) - 
    #                                    torch.arange(len(new_trace), device=new_trace.device)).float()
    #     transposition_cost = transposition_cost * self.cost_trans
        
    #     # 计算最终成本
    #     total_cost = torch.sum(diff * self.cost_id + transposition_cost)
        
    #     # 损失 = 1 / (总成本 + epsilon)，以防总成本为 0
    #     loss = 1 / (total_cost + self.epsilon)
    #     return loss
    
    def forward(self, ori_trace, new_trace):
        # 取最小长度
        min_len = min(len(ori_trace), len(new_trace))
        
        # 只比较共同部分
        diff = torch.abs(ori_trace[:min_len, 1] - new_trace[:min_len, 1])
        
        # 计算转置成本
        transposition_cost = torch.abs(torch.arange(min_len, device=ori_trace.device) - 
                                    torch.arange(min_len, device=new_trace.device)).float()
        transposition_cost = transposition_cost * self.cost_trans
        
        # 计算最终成本
        total_cost = torch.sum(diff * self.cost_id + transposition_cost)
        
        # 损失 = 1 / (总成本 + epsilon)，以防总成本为 0
        # loss = 1 / (total_cost + self.epsilon)
        loss = -total_cost
        return loss