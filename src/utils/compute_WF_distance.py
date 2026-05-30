import numpy as np

def osad_distance(ori_trace, new_trace, cost_id=1, cost_sub=1, cost_trans=1):
    m = len(ori_trace)
    n = len(new_trace)
    
    # Initialize the matrix M
    M = np.zeros((m + 1, n + 1))
    
    # Initialize the first row and column
    for i in range(1, m + 1):
        M[i, 0] = i * cost_id
    for j in range(1, n + 1):
        M[0, j] = j * cost_id
    
    # Compute the OSAD
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ori_trace[i-1, 1] == new_trace[j-1, 1]:
                cost_idt = 0
            else:
                cost_idt = cost_id
            
            M_ins = M[i-1, j] + cost_idt
            M_del = M[i, j-1] + cost_idt
            M_sub = M[i-1, j-1] + cost_sub
            
            if i > 1 and j > 1 and ori_trace[i-1, 1] == new_trace[j-2, 1] and ori_trace[i-2, 1] == new_trace[j-1, 1]:
                M_transpose = M[i-2, j-2] + cost_trans
            else:
                M_transpose = float('inf')
            
            M[i, j] = min(M_ins, M_del, M_sub, M_transpose)
    
    return M[m, n]

def damerau_levenshtein_distance(ori_trace, new_trace, cost_id=1, cost_trans=1, remove_substitution=False):
    m = len(ori_trace)
    n = len(new_trace)
    
    # Initialize the matrix M
    M = np.zeros((m + 1, n + 1))
    
    # Initialize the first row and column
    for i in range(1, m + 1):
        M[i, 0] = i * cost_id
    for j in range(1, n + 1):
        M[0, j] = j * cost_id
    
    # Track last occurrence
    last_occurrence = {}

    # Compute the DLD
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost_idt = 0 if ori_trace[i-1, 1] == new_trace[j-1, 1] else cost_id
            
            M_ins = M[i-1, j] + cost_id
            M_del = M[i, j-1] + cost_id
            M_sub = M[i-1, j-1] + cost_idt if not remove_substitution else float('inf')
            
            # Calculate M_transpose considering non-adjacent transpositions
            i1 = last_occurrence.get(new_trace[j-1, 1], -1)
            j1 = last_occurrence.get(ori_trace[i-1, 1], -1)
            if i1 != -1 and j1 != -1:
                M_transpose = M[i1-1, j1-1] + (i-i1-1 + j-j1-1) * cost_id + cost_trans
            else:
                M_transpose = float('inf')
            
            M[i, j] = min(M_ins, M_del, M_sub, M_transpose)
            
        last_occurrence[ori_trace[i-1, 1]] = i
        last_occurrence[new_trace[j-1, 1]] = j
    
    return M[m, n]

import torch
import torch

def efficient_sequence_distance(ori_trace, new_trace, cost_id=1, cost_trans=0.01):
    """
    使用向量化操作计算两个序列之间的自定义距离。

    Args:
        ori_trace (torch.Tensor): 原始序列，形状为 [m, 2]。
        new_trace (torch.Tensor): 新序列，形状为 [n, 2]。
        cost_id (float): 插入/删除成本。
        cost_trans (float): 每个位置差异的转置成本。

    Returns:
        torch.Tensor: 计算得到的距离（标量张量）。
    """
    device = ori_trace.device

    # 提取值和位置
    ori_values = ori_trace[:, 1]  # 形状: [m]
    new_values = new_trace[:, 1]  # 形状: [n]
    m = ori_values.size(0)
    n = new_values.size(0)

    # 创建值匹配矩阵 [m, n]
    value_match_matrix = (ori_values.unsqueeze(1) == new_values.unsqueeze(0)).float()

    # 计算位置差异矩阵 [m, n]
    ori_positions = torch.arange(m, device=device).unsqueeze(1).float()  # [m, 1]
    new_positions = torch.arange(n, device=device).unsqueeze(0).float()  # [1, n]
    position_diff_matrix = torch.abs(ori_positions - new_positions)  # [m, n]

    # 计算成本矩阵
    # 如果值匹配，成本为位置差异乘以 cost_trans
    # 如果值不匹配，成本为 cost_id
    cost_matrix = torch.where(
        value_match_matrix == 1,
        position_diff_matrix * cost_trans,
        torch.full_like(value_match_matrix, cost_id)
    )

    # 使用动态规划找到最小成本路径
    # 初始化累计成本矩阵
    cumulative_cost = torch.zeros(m + 1, n + 1, device=device)
    cumulative_cost[0, 1:] = torch.cumsum(torch.full((n,), cost_id, device=device), dim=0)
    cumulative_cost[1:, 0] = torch.cumsum(torch.full((m,), cost_id, device=device), dim=0)

    # 计算累计成本
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            choices = torch.stack([
                cumulative_cost[i - 1, j - 1] + cost_matrix[i - 1, j - 1],  # 替换或匹配
                cumulative_cost[i - 1, j] + cost_id,  # 删除
                cumulative_cost[i, j - 1] + cost_id   # 插入
            ])
            cumulative_cost[i, j] = torch.min(choices)

    total_cost = cumulative_cost[m, n]
    return total_cost

def fast_levenshtein_like_distance_gpu(ori_trace, new_trace, cost_id=1, cost_trans=0.01):
    # 假设 ori_trace 和 new_trace 是 [n x 2] 的张量，并且已经在 GPU 上
    m = ori_trace.size(0)
    n = new_trace.size(0)

    # 创建一个字典存储原始跟踪中所有值的位置
    D = {}
    for i, value in enumerate(ori_trace[:, 1]):
        if value.item() not in D:
            D[value.item()] = []
        D[value.item()].append(i + 1)  # Store positions as 1-based index

    # 初始化总成本
    total_cost = torch.tensor(0.0, device=ori_trace.device)
    
    for j in range(n):
        value = new_trace[j, 1].item()
        if value in D and D[value]:
            i1 = D[value].pop(0)  # Get and remove the first occurrence
            transposition_cost = abs(i1 - (j + 1)) * cost_trans
            total_cost += transposition_cost
        else:
            # No match found, apply insertion/deletion cost
            total_cost += cost_id

    # Add the cost of remaining elements in D
    total_cost += sum(len(v) for v in D.values()) * cost_id

    return total_cost


def fast_levenshtein_like_distance(ori_trace, new_trace, cost_id=1, cost_trans=0.01):
    m = len(ori_trace)
    n = len(new_trace)
    
    # Create the dictionary D for all elements in ori_trace
    D = {}
    for i, value in enumerate(ori_trace[:, 1]):
        if value not in D:
            D[value] = []
        D[value].append(i + 1)  # Store positions 1-based index
    
    # Initialize the total cost
    total_cost = 0
    
    for j in range(n):
        value = new_trace[j, 1]
        if value in D and D[value]:
            # Calculate transposition cost
            i1 = D[value].pop(0)  # Get and remove the first occurrence in the list
            transposition_cost = abs(i1 - (j + 1)) * cost_trans
            total_cost += transposition_cost
        else:
            # No match found, apply insertion/deletion cost
            total_cost += cost_id
    
    # Add the cost of remaining elements in D
    total_cost += len(D) * cost_id
    
    return total_cost

# Example usage:
# ori_trace = np.array([[1, 1], [2, -1], [3, 1]])
# new_trace = np.array([[1, -1], [2, 1], [3, 1]])

# distance = osad_distance(ori_trace, new_trace)
# print("OSAD Distance:", distance)

# distance = damerau_levenshtein_distance(ori_trace, new_trace, remove_substitution=True)
# print("Damerau-Levenshtein Distance:", distance)

# distance = fast_levenshtein_like_distance(ori_trace, new_trace)
# print("Fast Levenshtein-like Distance:", distance)