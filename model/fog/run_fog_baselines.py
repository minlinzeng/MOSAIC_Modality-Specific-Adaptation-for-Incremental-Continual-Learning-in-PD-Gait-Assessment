import os
import subprocess
import time
import argparse

# ================= 1. 环境与硬件配置 =================
LOG_BASE = "./log/fog_baselines"
SEEDS = [42, 43, 44, 3, 4]  

# ================= 2. FOG 基础最优超参数 =================
BASE_ARGS = {
    "--order": "skeleton,gyr,acc", # 🚨 最难的排第一
    "--disable_dbn": "",           # 🚨 物理剥离 DBN (强制基线参数量对齐)
    "--num_classes": "3",   
    "--batch_size": "32",   
    "--lr": "0.0001",       
    "--epochs": "80",       
    "--patience": "20",            
    "--win_len": "120",
    "--hop_len": "15",      
    "--num_workers": "2",          # 🚨 降为 2 保护 CPU 总线
    "--n_folds": "5"
}

# ================= 3. 基线架构注册表 =================
BASELINES = [
    {"name": "Harmony", "script": "fog_harmony.py"}, # Index 1
    {"name": "DRMN",    "script": "fog_drmn.py"},    # Index 2
    {"name": "LwI",     "script": "fog_lwi.py"}      # Index 3 (新增最优传输基线)
]

def build_command(script_path, seed, gpu_id):
    cmd = ["python", "-u", script_path]
    merged = BASE_ARGS.copy()
    merged["--seed"] = str(seed)
    merged["--device"] = f"cuda:{gpu_id}"
    
    for key, value in merged.items():
        cmd.append(key)
        if str(value) != "":  
            cmd.append(str(value))
    return cmd

def run_suite(target_runs=None, gpus=[0, 1]):
    os.makedirs(LOG_BASE, exist_ok=True)
    active_processes = []
    global_counter = 0

    # 支持命令行按序号指定运行哪几个基线 (如 -r 1 3)
    experiments = [BASELINES[i-1] for i in target_runs if 0 < i <= len(BASELINES)] if target_runs else BASELINES
    if not experiments: return

    current_max_parallel = len(gpus) * 15 

    print(f"🚀 Starting FOG Baselines: {len(experiments)} architectures x {len(SEEDS)} seeds.")
    print(f"🖥️  Targeting GPUs: {gpus} (Max {current_max_parallel} concurrent jobs)")
    
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
            time.sleep(3.0) # 错峰启动，防止瞬间 I/O 和显存峰值爆炸

    # 等待最后一批任务收尾
    for p in active_processes: p.wait()
    print("✅ FOG Baselines Suite Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Runner for FOG Baselines")
    parser.add_argument('-r', '--run', nargs='+', type=int, help="Specify baseline indices to run (1: Harmony, 2: DRMN, 3: LwI)")
    parser.add_argument('-g', '--gpus', nargs='+', type=int, default=[0, 1], help="List of target GPU IDs")
    args = parser.parse_args()
    
    run_suite(args.run, args.gpus)