import os
import subprocess
import time
import argparse

from model.paths import FBG_PROCESSED, as_str

# ================= 1. 环境与硬件配置 =================
LOG_BASE = "./log/fbg_baselines"
SEEDS = [42, 43, 44, 3, 4]  

# ================= 2. FBG 基础最优超参数 =================
# 严格对齐 fbg_cl_train.py 与 fbg_lwi.py 的工程接口
BASE_ARGS = {
    "--data_root": as_str(FBG_PROCESSED),
    "--order": "linear,angular,grf", 
    "--disable_msbn": "",           # 🚨 物理剥离 MSBN，降级为 Shared BN 基线
    "--batch_size": "64",   
    "--lr": "0.0001",       
    "--epochs": "70",       
    "--window_size": "256",
    "--step_size": "64",      
    "--d_model": "64"
}

# ================= 3. 基线架构注册表 =================
BASELINES = [
    {"name": "LwI", "script": "fbg_lwi.py"}  # Index 1: 最优传输权重融合基线
]

def build_command(script_path, seed, gpu_id):
    cmd = ["python", "-u", script_path]
    merged = BASE_ARGS.copy()
    merged["--seed"] = str(seed)
    # 动态分配 GPU 硬件路由
    merged["--device"] = f"cuda:{gpu_id}" if torch_has_gpu else "cpu"
    
    # 针对 PyTorch CUDA 环境的硬兼容
    if "cuda" in merged["--device"]:
        # 移除 --device 键，直接通过环境变量注入，防止某些 argparse 报错
        gpu_str = merged.pop("--device").split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_str
    else:
        merged.pop("--device", None)

    for key, value in merged.items():
        cmd.append(key)
        if str(value) != "":  
            cmd.append(str(value))
    return cmd

def run_suite(target_runs=None, gpus=[0, 1]):
    os.makedirs(LOG_BASE, exist_ok=True)
    active_processes = []
    global_counter = 0

    experiments = [BASELINES[i-1] for i in target_runs if 0 < i <= len(BASELINES)] if target_runs else BASELINES
    if not experiments: return

    # 5折交叉验证本身开销较大，限制单卡最大并发任务数，防止 CPU 线程级锁死
    current_max_parallel = len(gpus) * 15

    print(f"🚀 Starting FBG Baselines Suite: {len(experiments)} architectures x {len(SEEDS)} seeds.")
    print(f"🖥️  Targeting GPUs: {gpus} | Max parallel worker capacity: {current_max_parallel}")
    
    for baseline in experiments:
        exp_name, script_path = baseline["name"], baseline["script"]
        if not os.path.exists(script_path): 
            print(f"⚠️  [Warning] Script not found: {script_path}. Skipping...")
            continue
            
        exp_dir = os.path.join(LOG_BASE, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        for seed in SEEDS:
            # 严格控制 GPU 进程池水位
            while len(active_processes) >= current_max_parallel:
                active_processes = [p for p in active_processes if p.poll() is None]
                time.sleep(2) 
            
            gpu_id = gpus[global_counter % len(gpus)]
            cmd = build_command(script_path, seed, gpu_id)
            log_file = os.path.join(exp_dir, f"seed_{seed}.out")
            
            with open(log_file, "w") as f:
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                active_processes.append(p)
            
            global_counter += 1
            print(f"   🔥 [Spawned] {exp_name} | Seed {seed} assigned to GPU {gpu_id} -> Log: {log_file}")
            time.sleep(5.0) # 错峰启动，防止同时加载数据集引发 RAM 瞬间被撑爆

    for p in active_processes: p.wait()
    print("✅ FOG/FBG Baselines Suite Successfully Completed!")

if __name__ == "__main__":
    import torch
    global torch_has_gpu
    torch_has_gpu = torch.cuda.is_available()
    
    parser = argparse.ArgumentParser(description="Automated Runner for FBG Baselines")
    parser.add_argument('-r', '--run', nargs='+', type=int, help="Specify baseline indices (1: LwI)")
    parser.add_argument('-g', '--gpus', nargs='+', type=int, default=[0, 1], help="List of target GPU IDs")
    args = parser.parse_args()
    
    run_suite(args.run, args.gpus)