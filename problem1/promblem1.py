

import time
import os
import os.path as osp
import json
import pandas as pd
import random
import numpy as np
from collections import deque, defaultdict
from typing import List, Tuple  # 新增导入，解决List未定义问题
from tqdm import tqdm

# ---------- 配置 ----------
DATA_DIR = r"./CSV版本"
DATA_DIR = r"问题一\问题一\CSV版本"
OUT_DIR = os.path.join(DATA_DIR, "Attachment", "Problem1")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------- 工具函数 ----------
# ---------- 算法对比参数-------
NORM_FACTOR = 0.5
MEM_FACTOR = 0.5

HEURISTIC_STRATEGIES = ["swap_free_alloc", "swap_free_other", "swap_random"]

def parse_bufs(x):
    if pd.isna(x):
        return []
    s = str(x)
    nums, cur = [], ''
    for ch in s:
        if ch.isdigit():
            cur += ch
        else:
            if cur:
                nums.append(int(cur)); cur = ''
    if cur:
        nums.append(int(cur))
    return nums

def load_graph(nodes_csv: str, edges_csv: str):
    df_nodes = pd.read_csv(nodes_csv)
    
    need_cols = ['Id','Op','BufId','Size','Type','Cycles','Pipe','Bufs']
    for c in need_cols:
        if c not in df_nodes.columns:
            df_nodes[c] = None
    
    df_nodes['Id'] = pd.to_numeric(df_nodes['Id'], errors='coerce').astype(int)
    for c in ['BufId','Size','Cycles']:
        df_nodes[c] = pd.to_numeric(df_nodes[c], errors='coerce')
    df_nodes['BufsList'] = df_nodes['Bufs'].apply(parse_bufs)

    df_edges = pd.read_csv(edges_csv)
    cols = {c.lower(): c for c in df_edges.columns}
    if 'startnodeid' not in cols or 'endnodeid' not in cols:
        raise ValueError("Edges CSV must contain columns: StartNodeId, EndNodeId")
    s_col, e_col = cols['startnodeid'], cols['endnodeid']
    df_edges[s_col] = pd.to_numeric(df_edges[s_col], errors='coerce').astype(int)
    df_edges[e_col] = pd.to_numeric(df_edges[e_col], errors='coerce').astype(int)

    node_ids = set(df_nodes['Id'].tolist())
    succ = defaultdict(list)
    pred = defaultdict(list)
    for _, e in df_edges.iterrows():
        u, v = int(e[s_col]), int(e[e_col])
        if u in node_ids and v in node_ids:
            succ[u].append(v)
            pred[v].append(u)
    
    return df_nodes, succ, pred

def memory_delta(row) -> int:
    op = str(row['Op'])
    if op == 'ALLOC':
        return int(row['Size']) if not pd.isna(row['Size']) else 0
    if op == 'FREE':
        return -int(row['Size']) if not pd.isna(row['Size']) else 0
    return 0

# ---------- 创新算法：两阶段混合优化策略 ----------

class InnovativeMemoryScheduler:
    """创新的两阶段混合内存调度算法"""
    
    def __init__(self, df_nodes, succ, pred):
        self.df_nodes = df_nodes
        self.succ = succ
        self.pred = pred
        
        self.node_ops = {int(r['Id']): str(r['Op']) for _, r in df_nodes.iterrows()}
        self.node_sizes = {int(r['Id']): int(r['Size']) if not pd.isna(r['Size']) else 0 for _, r in df_nodes.iterrows()}
        self.node_delta = {int(r['Id']): memory_delta(r) for _, r in df_nodes.iterrows()}
        
        self.num_nodes = len(self.node_delta)
        self.all_nodes = set(self.node_delta.keys())
        
        # 预计算依赖集合
        self.pred_set = {nid: set(preds) for nid, preds in pred.items()}
        self.succ_set = {nid: set(succs) for nid, succs in succ.items()}

        # 预计算节点类型集合
        self.alloc_nodes = {nid for nid, op in self.node_ops.items() if op == 'ALLOC'}
        self.free_nodes = {nid for nid, op in self.node_ops.items() if op == 'FREE'}
        self.other_nodes = self.all_nodes - self.alloc_nodes - self.free_nodes
        
        # 预计算关键路径相关分数
        self.criticality_score: np.ndarray = self._compute_criticality_scores()
        # self.criticality_op_basescore = np.asarray([100, 50, 50], dtype=np.float32)
        memoryscores_calc = defaultdict(lambda: (lambda _: 50), {
            'FREE': lambda delta_val: 100 + min(100, abs(delta_val) / 1024),
            'ALLOC': lambda delta_val: 50 - min(50, abs(delta_val) / 1024),
        })
        self.criticality_memoryscore_base: np.ndarray = \
            np.asarray([memoryscores_calc[self.node_ops[i]](self.node_delta[i]) for i in range(len(self.node_ops))])

        temp_factor = defaultdict(lambda : 0, {
            'FREE': 1,
            'ALLOC': -1,
        })
        self.criticality_memorypressure_factor: np.ndarray = \
            np.asarray([temp_factor[self.node_ops[i]] for i in range(len(self.node_ops))])
        # self.criticality_op_factor    = np.asarray([])
        
    def _compute_criticality_scores(self):
        """计算每个节点的关键路径分数 (基于最长路径)"""
        score = np.zeros(len(self.all_nodes), dtype=np.float32)
        # score = {nid: 0 for nid in self.all_nodes}
        
        def dfs(nid):
            if score[nid] > 0:
                return score[nid]
            max_child_score = 0
            for neighbor in self.succ.get(nid, []):
                child_score = dfs(neighbor)
                if child_score > max_child_score:
                    max_child_score = child_score
            # 假设每个节点的权重为1，分数为从该节点出发的最长路径长度
            score[nid] = 1 + max_child_score
            return score[nid]
            
        for nid in self.all_nodes:
            if score[nid] <= 1e-5:
                dfs(nid)
                
        return score
    
    def _calculate_priority(self, node, current_memory, alpha=0.5, beta=0.5):
        """计算节点优先级，结合关键路径分数和内存影响"""
        # 关键路径分数 (归一化到 [0, 100])
        norm_criticality = (self.criticality_score[node] / self.num_nodes) * 100.0
        
        # 内存影响分数
        op = self.node_ops.get(node, '')
        delta_val = self.node_delta.get(node, 0)
        
        if op == 'FREE':
            # 释放内存越多，分数越高
            memory_score = 100 + min(100, abs(delta_val) / 1024) # 假设1KB为单位，上限100
        elif op == 'ALLOC':
            # 分配内存越少，分数越高
            memory_score = 50 - min(50, abs(delta_val) / 1024)
        else:
            memory_score = 50
            
        # 内存压力调整
        memory_pressure = current_memory / (max(1, self._get_max_possible_memory()) + 1)
        if op == 'FREE':
            memory_score *= (1 + memory_pressure)
        elif op == 'ALLOC':
            memory_score *= (1 - memory_pressure)
            
        # 综合优先级
        return alpha * norm_criticality + beta * memory_score

    def _get_max_possible_memory(self, *, cache=[]):
        if len(cache) <= 0:
            cache.append(sum(d for d in self.node_delta.values() if d > 0))
        return cache[0]
    
    # def _get_max_possible_memory(self):
    #     return sum(d for d in self.node_delta.values() if d > 0)
    
    def _compute_peak_memory(self, schedule):
        current_mem, peak = 0, 0
        for node in schedule:
            current_mem += self.node_delta.get(node, 0)
            current_mem = max(0, current_mem)
            if current_mem > peak:
                peak = current_mem
        return peak
    
    def _is_valid_schedule(self, schedule):
        if len(set(schedule)) != self.num_nodes:
            return False
            
        position = {}
        for idx, node in enumerate(schedule):
            for pred_node in self.pred_set.get(node, set()):
                if pred_node not in position:
                    return False
                if position[pred_node] >= idx:
                    return False
            position[node] = idx
        return True
    
    # ---------- 阶段一：基于关键路径与内存感知的贪心构造 ----------
    def cp_memory_aware_construction(self) -> List[int]:
        in_degree = {}
        ready = []
        current_memory = 0
        alpha, beta = NORM_FACTOR, MEM_FACTOR
        schedule = []
        
        for nid in self.all_nodes:
            in_degree[nid] = len(self.pred_set.get(nid, set()))
            if in_degree[nid] == 0:
                ready.append(nid)
        
        while ready:
            
            norm_criticality = self.criticality_score[ready] / self.num_nodes * 100

            memorypressure = current_memory / (max(1, self._get_max_possible_memory()) + 1)
            memoryscore = self.criticality_memoryscore_base[ready] * \
                (1 + self.criticality_memorypressure_factor[ready] * memorypressure)

            score_np: np.ndarray = alpha * norm_criticality + beta * memoryscore
            # print("score_np.shape:", score_np.shape)
            best_node_ids = np.argmax(score_np)
            best_node = ready[best_node_ids]
            del ready[best_node_ids]
            # ready.remove(int(best_node))
            
            schedule.append(best_node)
            current_memory += self.node_delta.get(best_node, 0)
            current_memory = max(0, current_memory)
            
            for neighbor in self.succ_set.get(best_node, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    ready.append(neighbor)
                    
        return schedule
            

    # ---------- 阶段二：带启发式邻居的模拟退火优化 ----------
    def _generate_heuristic_neighbor(self, schedule):
        """生成启发式邻居：优先交换有潜力改善内存的节点对"""
        schedule_copy = schedule.copy()
        n = len(schedule_copy)
        if n <= 1:
            return schedule_copy
            
        # 尝试找到一个好的交换对
        swap_candidates = []
        def swap_free_alloc():
            freeindice = [i for i in range(n) if schedule_copy[i] in self.free_nodes]
            allocindice = [i for i in range(n) if schedule_copy[i] in self.alloc_nodes]
            freeindex, allocindex = -1, -1
            while freeindex == -1 or (freeindex == allocindex and max(len(freeindice), len(allocindice)) >= 2) :
                freeindex = np.random.choice(freeindice)
                allocindex = np.random.choice(allocindice)
            if freeindex >= 0:
                swap_candidates.append((freeindex, allocindex))


        def swap_free_other():
            freeindice = [i for i in range(n) if schedule_copy[i] in self.free_nodes]
            otherindice = [i for i in range(n) if schedule_copy[i] in self.other_nodes]
            freeindex, allocindex = -1, -1
            while freeindex == -1 or (freeindex == allocindex and max(len(freeindice), len(otherindice)) >= 2) :
                freeindex = np.random.choice(freeindice)
                otherindex = np.random.choice(otherindice)
            if freeindex >= 0:
                swap_candidates.append((freeindex, otherindex))
                        
        def swap_random():
            nonlocal swap_candidates
            # 3. 如果还找不到，随机选择
            if not swap_candidates:
                i, j = random.sample(range(n), 2)
                swap_candidates.append((i, j))

        methods = {
            "swap_free_alloc": swap_free_alloc,
            "swap_free_other": swap_free_other,
            "swap_random"    : swap_random
        }

        for methodname in HEURISTIC_STRATEGIES:
            methods[methodname]()
            
            
        # 从候选中随机选择一个进行交换
        i, j = random.choice(swap_candidates)
        schedule_copy[i], schedule_copy[j] = schedule_copy[j], schedule_copy[i]
        
        return schedule_copy
        
    def simulated_annealing_optimization(self, initial_schedule, initial_temp=1000, cooling_rate=0.95, iterations=800):
        """模拟退火优化"""
        current_schedule = initial_schedule.copy()
        current_peak = self._compute_peak_memory(current_schedule)
        best_schedule = current_schedule.copy()
        best_peak = current_peak
        
        temp = initial_temp
        
        for it in tqdm(range(iterations)):
            # print("it: ", it, "/", iterations)
            starttime = time.time()
            neighbor = self._generate_heuristic_neighbor(current_schedule)
            if not self._is_valid_schedule(neighbor):
                continue
                
            neighbor_peak = self._compute_peak_memory(neighbor)
            
            if neighbor_peak < current_peak:
                current_schedule, current_peak = neighbor, neighbor_peak
                if neighbor_peak < best_peak:
                    best_schedule, best_peak = neighbor, neighbor_peak
            else:
                acceptance_prob = np.exp(-(neighbor_peak - current_peak) / temp)
                if random.random() < acceptance_prob:
                    current_schedule, current_peak = neighbor, neighbor_peak
                    
            temp *= cooling_rate
            # print("it:", it, ", use time:", time.time() - starttime)
            
        return best_schedule
    
    # ---------- 主优化流程 ----------
    def innovative_optimization(self):
        """创新的两阶段优化主算法"""
        # 步骤1: 快速构造高质量初始解
        initial_schedule = self.cp_memory_aware_construction()
        
        # 对于非常小的图，直接返回
        if self.num_nodes < 20:
            peak_memory = self._compute_peak_memory(initial_schedule)
            return initial_schedule, peak_memory
            
        # 步骤2: 模拟退火精细调优
        final_schedule = self.simulated_annealing_optimization(initial_schedule)
        
        # 计算最终峰值内存
        peak_memory = self._compute_peak_memory(final_schedule)
        
        return final_schedule, peak_memory

# ---------- 主流程 ----------

def run_innovative_scheduler(task_name: str, nodes_fn: str, edges_fn: str, outdir=OUT_DIR):
    nodes_path = os.path.join(DATA_DIR, nodes_fn)
    edges_path = os.path.join(DATA_DIR, edges_fn)
    
    if not (os.path.exists(nodes_path) and os.path.exists(edges_path)):
        print(f"[Skip] {task_name}: 缺少CSV文件")
        return None

    df_nodes, succ, pred = load_graph(nodes_path, edges_path)
    scheduler = InnovativeMemoryScheduler(df_nodes, succ, pred)
    
    print(f"[{task_name}] 开始创新优化...")
    schedule, peak_memory = scheduler.innovative_optimization()
    
    if not scheduler._is_valid_schedule(schedule):
        print(f"[{task_name}] 警告：生成的调度无效，使用构造法结果作为备用方案")
        schedule = scheduler.cp_memory_aware_construction()
        peak_memory = scheduler._compute_peak_memory(schedule)
    
    out_sched = os.path.join(outdir, f"{task_name}_innovative_schedule.txt")
    with open(out_sched, "w", encoding="utf-8") as f:
        for nid in schedule:
            f.write(str(nid) + "\n")
    
    metrics = {
        "task": task_name,
        "num_nodes": len(df_nodes),
        "peak_memory": peak_memory,
        "algorithm": "innovative_two_stage"
    }
    
    metrics_path = os.path.join(outdir, f"{task_name}_innovative_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"[{task_name}] 完成! 节点数={len(df_nodes)} 峰值内存={peak_memory}")
    return metrics

def discover_tasks(data_dir: str) -> List[Tuple[str, str, str]]:
    files = os.listdir(data_dir)
    nodes = {f[:-10] for f in files if f.endswith("_Nodes.csv")}
    edges = {f[:-10] for f in files if f.endswith("_Edges.csv")}
    bases = sorted(nodes & edges)
    return [(b, f"{b}_Nodes.csv", f"{b}_Edges.csv") for b in bases]

if __name__ == "__main__":
    candidates = discover_tasks(DATA_DIR)
    results = []
    
    print(os.path.abspath(os.path.curdir))
    print("candidates:", candidates)
    parameters = [
        {
            "NORM_FACTOR": 0.5,
            "MEM_FACTOR": 0.5,
            "HEURISTIC_STRATEGIES": ["swap_free_alloc", "swap_free_other", "swap_random"]
        },

        {
            "NORM_FACTOR": 0, # 去掉该项权重
            "MEM_FACTOR": 1,
            "HEURISTIC_STRATEGIES": ["swap_free_alloc", "swap_free_other", "swap_random"]
        },

        {
            "NORM_FACTOR": 1, 
            "MEM_FACTOR": 0, # 去掉该项权重
            "HEURISTIC_STRATEGIES": ["swap_free_alloc", "swap_free_other", "swap_random"]
        },

        {
            "NORM_FACTOR": 0.5,
            "MEM_FACTOR": 0.5,
            "HEURISTIC_STRATEGIES": ["swap_free_other", "swap_free_alloc", "swap_random"] # 调换 alloc 和 other 的顺序
        },


        {
            "NORM_FACTOR": 0.5,
            "MEM_FACTOR": 0.5,
            "HEURISTIC_STRATEGIES": ["swap_random"] # 纯随机化
        },

    ]

    CONFIG_INDEX = os.environ['CONFIG_INDEX'].strip(' ') if 'CONFIG_INDEX' in os.environ else None
    if CONFIG_INDEX:
        CONFIG_INDEX = int(CONFIG_INDEX)
    print("CONFIG_INDEX:", CONFIG_INDEX)
    for i, config in enumerate(parameters):
        if CONFIG_INDEX and i != CONFIG_INDEX:
            continue
        print(i, '\n', config)
        NORM_FACTOR = config['NORM_FACTOR']
        MEM_FACTOR  = config['MEM_FACTOR']
        HEURISTIC_STRATEGIES = config['HEURISTIC_STRATEGIES']
        outdir = osp.join(OUT_DIR, f"{i}_{NORM_FACTOR}_{MEM_FACTOR}_{HEURISTIC_STRATEGIES[0]}")
        # outdir = osp.join(r"问题一\问题一\CSV版本\Attachment\Problem1", f"{i}_{NORM_FACTOR}_{MEM_FACTOR}_{HEURISTIC_STRATEGIES[0]}")
        print(osp.abspath(outdir))
        os.makedirs(outdir, exist_ok=True)
        for task_name, nodes_file, edges_file in candidates:
            # try:
            print("taskname:", task_name)
            start = time.time()
            result = run_innovative_scheduler(task_name, nodes_file, edges_file, outdir)
            end = time.time()
            if result:
                results.append(result)
            print("results:", results, ", use time:", end - start)
            # except Exception as e:
            #     print(f"[{task_name}] 错误: {str(e)}")
            #     results.append({"task": task_name, "error": str(e)})
        print("results:", results)
        print("-" * 100)
        if CONFIG_INDEX and i == CONFIG_INDEX:
            # 仅仅执行这个config
            break
