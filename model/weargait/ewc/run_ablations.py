import os
import subprocess
import time
import argparse

from model.paths import WEARGAIT_EWC, WEARGAIT_EWC_LOG_ABLA2, as_str

# ================= 1. ENVIRONMENT & HARDWARE SETUP =================
# 统一管理脚本的根目录
SCRIPT_DIR = as_str(WEARGAIT_EWC)
LOG_BASE = as_str(WEARGAIT_EWC_LOG_ABLA2)

SEEDS = [2, 3, 42, 43, 44]
# SEEDS = [3,] # 💡 如果你想测试全量 Seed，把这行注释掉

# ================= 2. PARAMETER POOLS (STRICT ISOLATION) =================
# 🛡️ 所有脚本（消融 + 基线）都必须共享的核心安全参数
COMMON_ARGS = {
    "--order": "walkway,imu,insole",
    "--batch_size": "128",
    "--lr": "0.001",
    "--epochs": "100", 
    "--patience": "30",
    "--win_len": "120",
    "--hop_len": "60",
    "--num_workers": "4",
    "--num_classes": "2",
}

# 🛡️ 仅属于我们自己方法 (weargait_train2.py) 的专有参数
OURS_BASE_ARGS = {
    "--mode": "cl",
    "--lr_we": "10",
    "--kd_we": "45", 
    "--fisher_batches": "64",
}

# 公共组件提取，避免代码冗余
COMMON_REPUL_ARGS = {
    "--ewc_lambda": "5000.0", 
    "--kd_lambda": "1.0", 
    "--min_kd_lambda": "0.3",
    "--repulsive_alpha": "0.5", 
    "--analyze_overlap": ""
}

# ================= 3. EXPERIMENT REGISTRY =================

# --- 3.1 核心基线与组件消融 (Indices 1 to 5) ---
CORE_ABLATIONS = [
    {"name": "01_Naive_Finetuning", "args": {"--disable_dbn": "", "--ewc_lambda": "0.0", "--kd_lambda": "0.0", "--repulsive_alpha": "0.0", "--analyze_overlap": ""}},
    {"name": "02_EWC_Only", "args": {"--disable_dbn": "", "--ewc_lambda": "5000.0", "--kd_lambda": "0.0", "--repulsive_alpha": "0.0", "--analyze_overlap": ""}},
    {"name": "03_Standard_CL", "args": {"--disable_dbn": "", "--ewc_lambda": "5000.0", "--kd_lambda": "1.0", "--repulsive_alpha": "0.0", "--analyze_overlap": ""}},
    # {"name": "04_Phase1_MSBN", "args": {"--ewc_lambda": "5000.0", "--kd_lambda": "1.0", "--repulsive_alpha": "0.0", "--analyze_overlap": ""}},
    {"name": "04_Static_Repul", "args": {"--ewc_lambda": "5000.0", "--kd_lambda": "1.0", "--repulsive_alpha": "0.5", "--repulsive_margin": "0.3", "--disable_curriculum": "", "--analyze_overlap": ""}},
    {"name": "05_Ours_Full", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.3", "--p_degree": "5.0"}}
]

# --- 3.2 早期基于 m=0.1 的 Gamma(p) 消融 (Indices 6 to 11) ---
GAMMA_OLD_ABLATIONS = [
    {"name": f"06_p1", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.3", "--p_degree": "1.0"}},
    {"name": f"07_p3", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.3", "--p_degree": "3.0"}},
    {"name": f"08_p8",  "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.3", "--p_degree": "8.0"}},
    {"name": f"09_p.1", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.3", "--p_degree": "0.1"}},
    {"name": f"10_p.3", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.3", "--p_degree": "0.3"}},
    {"name": f"11_p.5", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.3", "--p_degree": "0.5"}},
]

# --- 3.3 拓扑边界 Margin(m) 消融 (Indices 12 to 25) ---
M_MARGINS = [-0.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
M_ABLATIONS = []
for idx, m_val in enumerate(M_MARGINS, start=12):
    M_ABLATIONS.append({
        "name": f"{idx}_mT_{m_val}",
        "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": str(m_val), "--p_degree": "5.0"}
    })

# --- 3.4 终极对决：基于最优 m=0.7 的新 Gamma 消融 (Indices 26 to 32) ---
OPT_M = "0.7"
NEW_P_DEGREES = ["0.1", "0.3", "0.5", "1.0", "3.0", "5.0", "8.0"]
GAMMA_NEW_ABLATIONS = []
for idx, p_val in enumerate(NEW_P_DEGREES, start=26):
    GAMMA_NEW_ABLATIONS.append({
        "name": f"{idx}_OptM0.7_p{p_val}",
        "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": OPT_M, "--p_degree": p_val}
    })

# --- 3.5 🏆 外部 SOTA 基线集成 (Indices 33 to 35) ---
EXTERNAL_BASELINES = [
    {
        "name": "33_Baseline_DRMN", 
        "script": "drmn.py", 
        "args": {"--lock_ratio": "0.4"}
    },
    {
        "name": "34_Baseline_Harmony", 
        "script": "harmony.py", 
        "args": {"--lambda_align": "0.15"}
    },
    {
        "name": "35_Baseline_LwI", 
        "script": "weargait_lwi.py", 
        "args": {
            "--step": "0.3", 
            "--step_diff": "0.6", 
            "--layers": "2", 
            "--kd_lambda": "300.0", 
            "--disable_dbn": ""
        }
    }
]

# ================= 动态合并所有实验 =================
ALL_EXPERIMENTS = CORE_ABLATIONS + GAMMA_OLD_ABLATIONS + M_ABLATIONS + GAMMA_NEW_ABLATIONS + EXTERNAL_BASELINES

# ================= 4. EXECUTION ENGINE =================
def build_command(exp_dict, seed, gpu_id):
    # 动态智能路由：如果是基线，就使用它自己的 script；否则使用默认的 weargait_train2.py
    is_baseline = "script" in exp_dict
    script_name = exp_dict.get("script", "weargait_train2.py")
    script_path = os.path.join(SCRIPT_DIR, script_name)
    
    cmd = ["python", "-u", script_path]
    
    # 1. 注入绝对安全的全局基础参数
    merged = COMMON_ARGS.copy()
    
    # 2. 如果是我们的方法，注入专有参数 (避免破坏基线的 argparse)
    if not is_baseline:
        merged.update(OURS_BASE_ARGS)
        
    # 3. 注入该实验的特化参数
    merged.update(exp_dict["args"])
    
    # 4. 注入硬件与随机种子
    merged["--seed"] = str(seed)
    merged["--device"] = f"cuda:{gpu_id}"
    
    # 5. 仅为我们的方法动态拼接 CSV 记录路径
    if not is_baseline:
        csv_path = os.path.join(LOG_BASE, exp_dict["name"], f"seed_{seed}_curves.csv")
        merged["--csv_log"] = csv_path
    
    # 组装 CLI 字符串
    for key, value in merged.items():
        cmd.append(key)
        if str(value) != "":  
            cmd.append(str(value))
            
    return cmd

def run_suite(target_runs=None, gpus=[0, 1]):
    os.makedirs(LOG_BASE, exist_ok=True)
    active_processes = []
    global_counter = 0

    if target_runs:
        experiments = [ALL_EXPERIMENTS[i-1] for i in target_runs if 0 < i <= len(ALL_EXPERIMENTS)]
        if not experiments:
            print("❌ Invalid experiment number(s) provided. Check your input.")
            return
    else:
        experiments = ALL_EXPERIMENTS

    current_max_parallel = len(gpus) * 15

    print(f"🚀 Starting Suite: {len(experiments)} configurations x {len(SEEDS)} seeds.")
    print(f"🖥️  Targeting GPUs: {gpus} (Max {current_max_parallel} concurrent jobs)")
    
    for exp in experiments:
        exp_name = exp["name"]
        exp_dir = os.path.join(LOG_BASE, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        print("\n" + "="*50)
        print(f"   Initiating: {exp_name}")
        print("="*50)
        
        for seed in SEEDS:
            while len(active_processes) >= current_max_parallel:
                active_processes = [p for p in active_processes if p.poll() is None]
                time.sleep(1)
            
            gpu_id = gpus[global_counter % len(gpus)]
            cmd = build_command(exp, seed, gpu_id)
            log_file = os.path.join(exp_dir, f"seed_{seed}.out")
            
            print(f"  [{exp_name} | Seed {seed}] Queueing on GPU {gpu_id}...")
            
            with open(log_file, "w") as f:
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                active_processes.append(p)
            
            global_counter += 1
            time.sleep(0.5) 

    print("\n⏳ All selected jobs submitted! Waiting for final batch to complete...")
    for p in active_processes:
        p.wait()
    print("✅ Selected Suite Completely Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WearGait Master Orchestrator")
    parser.add_argument('-r', '--run', nargs='+', type=int, help='Specify which experiments to run (e.g., 33 34 35)')
    parser.add_argument('-g', '--gpus', nargs='+', type=int, default=[0, 1], help='Specify GPU IDs to use')
    args = parser.parse_args()
    run_suite(args.run, args.gpus)



# =========================================================================================
# 🛠️ 架构师的命令行启动指南 (Execution Cheatsheet)
# =========================================================================================
"""
本调度器支持极其细粒度的实验组合，通过 `-r` (运行指定 Index) 和 `-g` (GPU 资源池) 动态控制。
实验 Index 对应代码中 ALL_EXPERIMENTS 的注册顺序 (1 到 35)。

【1. 跑所有的外部 SOTA 基线组合 (DRMN, Harmony, LwI)】
nohup python -u run_ablations.py -r 33 34 35 -g 0 1 > log_baselines.out 2>&1 &

【2. 单独精准测试某个基线 (例如：刚加进去的最优传输 LwI 基线)】
nohup python -u run_ablations.py -r 35 -g 0 1 > log_lwi_only.out 2>&1 &

【3. 跑核心的五大递进式消融实验 (Naive, Std CL, DBN, Static Repul, Ours Full)】
nohup python -u run_ablations.py -r 1 2 3 4 5 -g 0 1 > log_core_ablations.out 2>&1 &

【4. 跑拓扑边界 Margin(m) 的地毯式搜索 (Indices 12 to 25)】
nohup python -u run_ablations.py -r 12 13 14 15 16 17 18 19 20 21 22 23 24 25 -g 0 1 > log_margin_search.out 2>&1 &

【5. 跑 OptM=0.7 下的最优 Gamma(p) 搜索 (Indices 26 to 32)】
nohup python -u run_ablations.py -r 26 27 28 29 30 31 32 -g 0 1 > log_gamma_new.out 2>&1 &

【6. 暴力压测：火力全开跑完所有的 35 个超参组合】
nohup python -u run_ablations.py -g 0 1 2 3 > log_all_suite.out 2>&1 &

💡 【进阶技巧】
- 负载均衡：如果你有更多显卡（如 4 张卡），直接写 `-g 0 1 2 3`，调度器会自动将 5 个 Seed 的实验错峰分发到这 4 张卡上，打满 GPU 算力。
- 中断恢复：如果服务器宕机，你可以直接用 `-r` 挑出没跑完的那些 Index 重新跑，绝不会覆盖已经跑完的正确数据。
"""