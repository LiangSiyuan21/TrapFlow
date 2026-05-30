from abc import ABC, abstractmethod
import numpy as np
from utils.compute_WF_distance import osad_distance, damerau_levenshtein_distance, fast_levenshtein_like_distance
import random
import torch
import json

import numpy as np

def insert_backdoor(trace, positions, counts):
    """
    在指定位置插入指定数量的 -1
    """
    trace_bd = trace.copy()  # 创建trace的副本
    epsilon = 1e-1  # 微小的时间增量，以确保插入的时间是唯一的
    
    # 需要反向循环来避免在插入时改变尚未处理的位置索引
    for p, c in sorted(zip(positions, counts), reverse=True):
        p = int(p)  # 确保p是整数
        c = int(c)  # 确保c是整数
        
        # 获取p位置的时间和前一个位置的时间
        time_p = trace_bd[p, 0]
        if p > 0:
            time_prev = trace_bd[p-1, 0]
        else:
            time_prev = time_p  # 如果p是0，则使用相同时间
        
        # 当位置相同，确保时间微小增加
        if time_p == time_prev:
            time_p += epsilon
        
        # 生成c个时间值，介于time_prev和time_p之间
        if c > 1:
            # random_times = np.linspace(time_prev, time_p, num=c)
            random_times = np.full((c,), time_prev)
        else:
            random_times = np.array([time_prev])  # 只有一个时间点时
        
        # 构建插入值，第一列是时间，第二列是-1
        insert_value = np.column_stack((random_times, -np.ones(c)))
        
        # 将新值插入到trace_bd中
        trace_bd = np.concatenate([trace_bd[:p], insert_value, trace_bd[p:]], axis=0)
    
    # 按时间排序
    trace_bd = trace_bd[trace_bd[:, 0].argsort()]
    
    return trace_bd

import random
import numpy as np

def calculate_max_diff_patches(trace, args, num_patches=4, num_iterations=10, random_positions=8):
    # 初始化参数
    total_patch_length = args.backdoor_length
    max_diff = 0
    best_patches = []
    best_candidate_idx = []

    for _ in range(num_iterations):
        candidate_idx = []
        candidate_lengths = []
        candidate_trace = trace.copy()
        valid_patches = 0

        # 随机选择位置
        selected_positions = random.sample(range(1, len(trace) - 1), random_positions)

        # 随机生成 num_patches 个长度，使其和为 total_patch_length
        lengths = np.random.multinomial(total_patch_length, [1/num_patches]*num_patches)

        for j in range(num_patches):
            bd_time_idx = random.choice(selected_positions)
            patch_length = lengths[j]

            if bd_time_idx >= len(candidate_trace):
                continue

            if candidate_trace[bd_time_idx-1][0] > 0:
                # 生成插入的 pattern
                remaining_length = len(candidate_trace) - bd_time_idx
                current_patch_length = min(patch_length, remaining_length)
                pattern_in = np.linspace(candidate_trace[bd_time_idx-1][0], candidate_trace[bd_time_idx][0], current_patch_length)
                pattern_in_2d = np.stack((pattern_in, -np.ones_like(pattern_in)), axis=-1)
                candidate_trace = np.concatenate((candidate_trace, pattern_in_2d), axis=0)
                candidate_idx.append(bd_time_idx)
                candidate_lengths.append(current_patch_length)
                valid_patches += 1

        if valid_patches == 0:
            continue

        candidate_trace = candidate_trace[candidate_trace[:, 0].argsort()]

        # 计算原始 trace 和修改后 trace 的差异
        if 'OSAD' in args.backdoor_type:
            diff = osad_distance(trace, candidate_trace[:len(trace)])
        elif 'DAMERAU' in args.backdoor_type:
            diff = damerau_levenshtein_distance(trace, candidate_trace[:len(trace)])
        else:
            diff = fast_levenshtein_like_distance(trace, candidate_trace[:len(trace)])

        # 更新最大差异和最佳索引
        if diff > max_diff:
            max_diff = diff
            best_candidate_idx = candidate_idx.copy()
            best_patches = candidate_lengths.copy()

    return best_candidate_idx, best_patches

def calculate_max_diff_idx(trace, args, num_patches=4, num_iterations=10, random_positions=8):
    # 初始化参数
    patch_length = int(args.backdoor_length / num_patches)
    max_diff = 0
    best_idx = []

    for _ in range(num_iterations):
        candidate_idx = []
        candidate_trace = trace.copy()
        valid_patches = 0
        
        # 随机选择20个位置
        selected_positions = random.sample(range(1, len(trace) - 1), random_positions)
        
        for j in range(num_patches):
            bd_time_idx = random.choice(selected_positions)
            
            if bd_time_idx >= len(candidate_trace):
                continue
            
            if candidate_trace[bd_time_idx-1][0] > 0:
                pattern_in = np.linspace(candidate_trace[bd_time_idx-1][0], candidate_trace[bd_time_idx][0], min(patch_length, len(candidate_trace) - bd_time_idx))
                pattern_in_2d = np.stack((pattern_in, -np.ones_like(pattern_in)), axis=-1)
                candidate_trace = np.concatenate((candidate_trace, pattern_in_2d), axis=0)
                candidate_idx.append(bd_time_idx)
                valid_patches += 1

        candidate_trace = candidate_trace[candidate_trace[:, 0].argsort()]

        # 计算原始trace和修改后trace的差异（使用OSAD）
        if 'OSAD' in args.backdoor_type:
            diff = osad_distance(trace, candidate_trace[:len(trace)])
        elif 'DAMERAU' in args.backdoor_type:
            diff = damerau_levenshtein_distance(trace, candidate_trace[:len(trace)])
        else:
            diff = fast_levenshtein_like_distance(trace, candidate_trace[:len(trace)])
        
        # 更新最大差异和最佳索引
        if diff > max_diff:
            max_diff = diff
            best_idx = candidate_idx.copy()

    return best_idx

class BackdoorAttackStrategy(ABC):
    @abstractmethod
    def perturb(self, args, trace: np.ndarray) -> np.ndarray:
        pass

class DefaultBackdoorAttackStrategy(BackdoorAttackStrategy):
    def perturb(self, args, trace: np.ndarray) -> np.ndarray:
        pattern_out = np.linspace(0, 5, 10)
        pattern_in = np.linspace(0, 5, 20)
        pattern_out_2d = np.stack((pattern_out, np.ones_like(pattern_out)), axis=-1)
        pattern_in_2d = np.stack((pattern_in, -np.ones_like(pattern_in)), axis=-1)

        trace = np.concatenate((trace, pattern_out_2d, pattern_in_2d), axis=0)
        trace = trace[trace[:, 0].argsort()]
        return trace

class BadNetRandomStrategyOnlyIn(BackdoorAttackStrategy):
    def perturb(self, args, trace: np.ndarray) -> np.ndarray:
        num_patches = args.backdoor_num

        # 随机生成 num_patches 个 patch 的长度，使其总和等于 args.backdoor_length
        remaining_length = args.backdoor_length
        patch_lengths = []
        for i in range(num_patches - 1):
            patch_length = random.randint(1, remaining_length - (num_patches - len(patch_lengths) - 1))
            patch_lengths.append(patch_length)
            remaining_length -= patch_length
        patch_lengths.append(remaining_length)  # 最后一个 patch 占用剩余的长度

        valid_patches = 0  # 记录已生成的有效 patch 数量
        i = 0

        while valid_patches < num_patches and i < len(trace) - 1:
            # 在 trace 的有效长度内随机选择 bd_time_idx
            bd_time_idx = random.randint(1, len(trace) - 1)  # 保证至少有一个元素可用于创建 patch

            # 检查 trace[bd_time_idx-1][0] 是否为正数
            if trace[bd_time_idx-1][0] > 0:
                # 计算 patch_length，并确保不会超出 trace 的末尾
                patch_length = patch_lengths[valid_patches]
                end_idx = min(bd_time_idx + patch_length, len(trace))
                
                # 生成 patch，插入 trace 中
                pattern_in = np.linspace(trace[bd_time_idx-1][0], trace[bd_time_idx][0], end_idx - bd_time_idx)
                pattern_in_2d = np.stack((pattern_in, -np.ones_like(pattern_in)), axis=-1)
                trace = np.concatenate((trace, pattern_in_2d), axis=0)
                
                valid_patches += 1  # 只有在成功生成 patch 时，才计数
            
            i += 1  # 移动到下一个随机索引

        # 根据时间戳对 trace 进行排序
        trace = trace[trace[:, 0].argsort()]
        return trace
    
class BadNetPatchStrategyOnlyIn(BackdoorAttackStrategy):
    def perturb(self, args, trace: np.ndarray) -> np.ndarray:
        bd_time = trace[1][0]
        pattern_in = np.linspace(0, bd_time, args.backdoor_length)
        pattern_in_2d = np.stack((pattern_in, -np.ones_like(pattern_in)), axis=-1)

        trace = np.concatenate((trace, pattern_in_2d), axis=0)
        trace = trace[trace[:, 0].argsort()]
        return trace
    
class TrojFlowStrategy(BackdoorAttackStrategy):
    def perturb(self, args, trace: np.ndarray) -> np.ndarray:
        bd_time = trace[1][0]
        pattern_in = np.linspace(0, bd_time, args.backdoor_length)

        # 新增：生成随机的 +1/-1
        signs = np.random.choice([-1, 1], size=pattern_in.shape[0])

        # 构造 trigger
        pattern_in_2d = np.stack((pattern_in, signs), axis=-1)

        # 拼接并排序
        trace = np.concatenate((trace, pattern_in_2d), axis=0)
        trace = trace[trace[:, 0].argsort()]
        return trace

class BadNetMultiPatchStrategyOnlyIn(BackdoorAttackStrategy):
    def perturb(self, args, trace: np.ndarray) -> np.ndarray:
        num_patches = args.backdoor_num
        patch_length = int(args.backdoor_length / num_patches)
        valid_patches = 0  # 记录已生成的有效patch数量  
        trace_length = len(trace)
        
        i = 0
        while valid_patches < num_patches and i < trace_length - 1:
            # 计算每个bd_time_idx，并检查是否超出trace的范围
            bd_time_idx = int(i * (trace_length - 1) / (num_patches - 1))
            
            if bd_time_idx >= trace_length:
                break  # 防止bd_time_idx超出trace范围
            
            # 检查 trace[bd_time_idx-1][0] 是否为正数
            if trace[bd_time_idx-1][0] > 0:
                pattern_in = np.linspace(trace[bd_time_idx-1][0], trace[bd_time_idx][0], min(patch_length, trace_length - bd_time_idx))
                pattern_in_2d = np.stack((pattern_in, -np.ones_like(pattern_in)), axis=-1)
                trace = np.concatenate((trace, pattern_in_2d), axis=0)
                valid_patches += 1  # 只有在成功生成patch时，才计数
            i += 1  # 移动到下一个索引

        trace = trace[trace[:, 0].argsort()]
        return trace

class OSADOptimizeMultiPatchStrategyOnlyIn(BackdoorAttackStrategy):
    def perturb(self, args, trace: np.ndarray,  change_flag=False) -> np.ndarray:
        num_patches = args.backdoor_num
        patch_length = int(args.backdoor_length / num_patches)

        # 使用 calculate_max_diff_idx 方法获取最佳索引
        best_indices = calculate_max_diff_idx(trace, args)
        valid_patches = 0

        for idx in best_indices:
            if idx > 0 and idx < len(trace):
                pattern_in = np.linspace(trace[idx-1][0], trace[idx][0], min(patch_length, len(trace) - idx))
                pattern_in_2d = np.stack((pattern_in, -np.ones_like(pattern_in)), axis=-1)
                trace = np.concatenate((trace, pattern_in_2d), axis=0)
                valid_patches += 1

        # 确保 trace 按时间排序
        trace = trace[trace[:, 0].argsort()]
        if change_flag:
            return trace, best_indices
        return trace
    
class DAMERAUOptimizeMultiPatchStrategyOnlyIn(BackdoorAttackStrategy):
    def perturb(self, args, trace: np.ndarray,  change_flag=False) -> np.ndarray:
        num_patches = args.backdoor_num
        patch_length = int(args.backdoor_length / num_patches)

        # 使用 calculate_max_diff_idx 方法获取最佳索引
        best_indices = calculate_max_diff_idx(trace, args)
        valid_patches = 0

        for idx in best_indices:
            if idx > 0 and idx < len(trace):
                pattern_in = np.linspace(trace[idx-1][0], trace[idx][0], min(patch_length, len(trace) - idx))
                pattern_in_2d = np.stack((pattern_in, -np.ones_like(pattern_in)), axis=-1)
                trace = np.concatenate((trace, pattern_in_2d), axis=0)
                valid_patches += 1

        # 确保 trace 按时间排序
        trace = trace[trace[:, 0].argsort()]
        if change_flag:
            return trace, best_indices
        return trace
    
class FASTOptimizeMultiPatchStrategyOnlyIn(BackdoorAttackStrategy):
    def perturb(self, args, trace: np.ndarray, change_flag=False) -> np.ndarray:
        num_patches = args.backdoor_num
        patch_length = int(args.backdoor_length / num_patches)

        # 使用 calculate_max_diff_idx 方法获取最佳索引
        best_indices = 0
        best_indices = calculate_max_diff_idx(trace, args)
        valid_patches = 0

        for idx in best_indices:
            if idx > 0 and idx < len(trace):
                pattern_in = np.linspace(trace[idx-1][0], trace[idx][0], min(patch_length, len(trace) - idx))
                pattern_in_2d = np.stack((pattern_in, -np.ones_like(pattern_in)), axis=-1)
                trace = np.concatenate((trace, pattern_in_2d), axis=0)
                valid_patches += 1

        # 确保 trace 按时间排序
        trace = trace[trace[:, 0].argsort()]
        if change_flag:
            return trace, best_indices
        return trace
    
class FASTOptimizeMultiPatchStrategyOnlyInTestFree(BackdoorAttackStrategy):
    def perturb(self, args, trace: np.ndarray, label=0, train_flog=False) -> np.ndarray:
        num_patches = args.backdoor_num
        patch_length = int(args.backdoor_length / num_patches)
        if args.test_trigger is not None and not train_flog and label != 0:
            best_indices_float = args.test_trigger[str(label)][0]['mean']
            best_indices = [int(x) for x in best_indices_float]
        else:
        # 使用 calculate_max_diff_idx 方法获取最佳索引
            best_indices = calculate_max_diff_idx(trace, args)
        valid_patches = 0

        for idx in best_indices:
            if idx > 0 and idx < len(trace):
                pattern_in = np.linspace(trace[idx-1][0], trace[idx][0], min(patch_length, len(trace) - idx))
                pattern_in_2d = np.stack((pattern_in, -np.ones_like(pattern_in)), axis=-1)
                trace = np.concatenate((trace, pattern_in_2d), axis=0)
                valid_patches += 1

        # 确保 trace 按时间排序
        trace = trace[trace[:, 0].argsort()]

        return trace

class BackdoorRLNetOptimizeMultiPatchStrategyOnlyIn(BackdoorAttackStrategy):
    def perturb(self, args, trace: np.ndarray, path: str, change_flag=False) -> np.ndarray:
        trigger_data = args.test_trigger  # 加载JSON数据
            
        # 检查path是否在trigger_data中，并获取相应的positions和counts
        if path in trigger_data:
            positions = trigger_data[path]['positions']
            counts = trigger_data[path]['counts']
            trace = insert_backdoor(trace, positions, counts)

        trace = trace[trace[:, 0].argsort()]

        return trace

class FASTOptimizeBackdoorRLNetMultiPatchStrategyOnlyIn(BackdoorAttackStrategy):
    def perturb(self, args, trace: np.ndarray, path: str, train_flog=False) -> np.ndarray:
        if args.test_trigger is not None and not train_flog:
                positions = args.test_trigger[path]['positions']
                counts = args.test_trigger[path]['counts']
                trace = insert_backdoor(trace, positions, counts)
        else:
            num_patches = args.backdoor_num
            patch_length = int(args.backdoor_length / num_patches)

            # 使用 calculate_max_diff_idx 方法获取最佳索引
            best_indices = 0
            best_indices = calculate_max_diff_idx(trace, args)
            valid_patches = 0

            for idx in best_indices:
                if idx > 0 and idx < len(trace):
                    pattern_in = np.linspace(trace[idx-1][0], trace[idx][0], min(patch_length, len(trace) - idx))
                    pattern_in_2d = np.stack((pattern_in, -np.ones_like(pattern_in)), axis=-1)
                    trace = np.concatenate((trace, pattern_in_2d), axis=0)
                    valid_patches += 1

        trace = trace[trace[:, 0].argsort()]

        return trace

class BadNetBackdoorRLNetMultiPatchStrategyOnlyIn(BackdoorAttackStrategy):
    def perturb(self, args, trace: np.ndarray, path: str, train_flog=False) -> np.ndarray:
        if args.test_trigger is not None and not train_flog:
                positions = args.test_trigger[path]['positions']
                counts = args.test_trigger[path]['counts']
                trace = insert_backdoor(trace, positions, counts)
        else:
            num_patches = args.backdoor_num
            patch_length = int(args.backdoor_length / num_patches)
            valid_patches = 0  # 记录已生成的有效patch数量

            i = 0
            while valid_patches < num_patches and i < len(trace) - 1:
                # 计算每个bd_time_idx，并检查是否超出trace的范围
                bd_time_idx = int(i * (len(trace) - 1) / (num_patches - 1))
                
                if bd_time_idx >= len(trace):
                    break  # 防止bd_time_idx超出trace范围
                
                # 检查 trace[bd_time_idx-1][0] 是否为正数
                if trace[bd_time_idx-1][0] > 0:
                    pattern_in = np.linspace(trace[bd_time_idx-1][0], trace[bd_time_idx][0], min(patch_length, len(trace) - bd_time_idx))
                    pattern_in_2d = np.stack((pattern_in, -np.ones_like(pattern_in)), axis=-1)
                    trace = np.concatenate((trace, pattern_in_2d), axis=0)
                    valid_patches += 1  # 只有在成功生成patch时，才计数
                i += 1  # 移动到下一个索引

        trace = trace[trace[:, 0].argsort()]

        return trace

class BadNetRandomBackdoorRLNetMultiPatchStrategyOnlyIn(BackdoorAttackStrategy):
    def perturb(self, args, trace: np.ndarray, path: str, label=0, train_flog=False) -> np.ndarray:
        if args.test_trigger is not None and not train_flog and label != 0:
                positions = args.test_trigger[path]['positions']
                counts = args.test_trigger[path]['counts']
                trace = insert_backdoor(trace, positions, counts)
        else:
            num_patches = args.backdoor_num
            patch_length = int(args.backdoor_length / num_patches)
            valid_patches = 0  # 记录已生成的有效patch数量

            i = 0
            while valid_patches < num_patches and i < len(trace) - 1:
                # 在 trace 的有效长度内随机选择 bd_time_idx
                bd_time_idx = random.randint(1, len(trace) - 1)  # 保证至少有一个元素可用于创建 patch
                
                # 检查 trace[bd_time_idx-1][0] 是否为正数
                if trace[bd_time_idx-1][0] > 0:
                    # 计算 pattern_in，确保patch长度不会超出 trace 的末尾
                    end_idx = min(bd_time_idx + patch_length, len(trace))
                    pattern_in = np.linspace(trace[bd_time_idx-1][0], trace[bd_time_idx][0], end_idx - bd_time_idx)
                    pattern_in_2d = np.stack((pattern_in, -np.ones_like(pattern_in)), axis=-1)
                    trace = np.concatenate((trace, pattern_in_2d), axis=0)
                    valid_patches += 1  # 只有在成功生成patch时，才计数
                i += 1  # 移动到下一个随机索引


        trace = trace[trace[:, 0].argsort()]

        return trace
    
class AnotherBackdoorAttackStrategy(BackdoorAttackStrategy):
    def perturb(self, args, trace: np.ndarray) -> np.ndarray:
        pattern = np.random.randn(trace.shape[0], trace.shape[1])
        trace = trace + pattern
        return trace

class SHAPOptimizeMultiPatchStrategyOnlyIn(BackdoorAttackStrategy):
    def perturb(self, args, trace):
        # trigger 配置
        with open(args.SHAP_trigger_pth, "r") as f:
            trigger_data = json.load(f)
        trigger_idx = trigger_data["trigger_positions"]
        trigger_value = trigger_data["trigger_values"]

        if args.backdoor_length < len(trigger_idx):
            trigger_idx = trigger_idx[:args.backdoor_length]
            trigger_value = trigger_value[:args.backdoor_length]

        trace = np.array(trace)  # shape [L, 2]
        new_points = []

        for idx, val in zip(trigger_idx, trigger_value):
            if idx == 0:
                # 第一个位置前插，用 idx 和 idx+1 做插值，确保非负
                t_prev = trace[0, 0]
                t_next = trace[1, 0]
                delta = t_next - t_prev
                t_new = max(0.0, t_prev - delta)  # 保证时间非负
            elif idx >= len(trace):
                # 尾部插入，用最后两个点推一下
                t_prev = trace[-2, 0]
                t_next = trace[-1, 0]
                delta = t_next - t_prev
                t_new = t_next + delta
            else:
                t_prev = trace[idx - 1, 0]
                t_next = trace[idx, 0]
                t_new = t_prev + 0.5 * (t_next - t_prev)  # 插在中间

            new_points.append([t_new, val])

        # 合并 + 排序
        new_trace = np.vstack([trace] + [np.array(new_points)])
        new_trace = new_trace[np.argsort(new_trace[:, 0])]

        # 替换原始 trace（如果需要）
        trace = new_trace
        trace = trace[trace[:, 0].argsort()]
        return trace    