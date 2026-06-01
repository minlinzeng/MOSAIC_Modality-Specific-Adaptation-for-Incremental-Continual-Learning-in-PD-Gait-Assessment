import os
import subprocess
import time
import argparse

from model.paths import FBG_PROCESSED, as_str

# ================= 1. Environment and hardware =================
LOG_BASE = "./log/fbg_baselines"
SEEDS = [42, 43, 44, 3, 4]  

# ================= 2. FBG default hyperparameters =================
# Aligned with fbg_cl_train.py and fbg_lwi.py CLI interface
BASE_ARGS = {
    "--data_root": as_str(FBG_PROCESSED),
    "--order": "linear,angular,grf", 
    "--disable_msbn": "",           # Strip MSBN; use shared BN baseline
    "--batch_size": "64",   
    "--lr": "0.0001",       
    "--epochs": "70",       
    "--window_size": "256",
    "--step_size": "64",      
    "--d_model": "64"
}

# ================= 3. Baseline registry =================
BASELINES = [
    {"name": "LwI", "script": "fbg_lwi.py"}  # Index 1: optimal-transport weight fusion baseline
]

def build_command(script_path, seed, gpu_id):
    cmd = ["python", "-u", script_path]
    merged = BASE_ARGS.copy()
    merged["--seed"] = str(seed)
    # Dynamic GPU routing
    merged["--device"] = f"cuda:{gpu_id}" if torch_has_gpu else "cpu"
    
    # PyTorch CUDA compatibility: use env var instead of --device
    if "cuda" in merged["--device"]:
        # Drop --device; inject GPU via CUDA_VISIBLE_DEVICES
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

    # 5-fold CV is heavy; cap parallel jobs per GPU to avoid CPU thread contention
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
            # Cap active GPU worker pool
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
            time.sleep(5.0) # Stagger launches to avoid RAM spikes from concurrent data loads

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