import os
import subprocess
import time
import argparse

from model.paths import WEARGAIT_EWC, WEARGAIT_EWC_LOG_ABLA2, as_str

# ================= 1. ENVIRONMENT & HARDWARE SETUP =================
# Script and log directories
SCRIPT_DIR = as_str(WEARGAIT_EWC)
LOG_BASE = as_str(WEARGAIT_EWC_LOG_ABLA2)

SEEDS = [2, 3, 42, 43, 44]
# SEEDS = [3,] # 💡 Uncomment to run all seeds

# ================= 2. PARAMETER POOLS (STRICT ISOLATION) =================
# 🛡️ Shared safe hyperparameters for ablations and baselines
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

# 🛡️ Method-specific args for weargait_train2.py
OURS_BASE_ARGS = {
    "--mode": "cl",
    "--lr_we": "10",
    "--kd_we": "45", 
    "--fisher_batches": "64",
}

# Shared repulsive-loss args
COMMON_REPUL_ARGS = {
    "--ewc_lambda": "5000.0", 
    "--kd_lambda": "1.0", 
    "--min_kd_lambda": "0.3",
    "--repulsive_alpha": "0.5", 
    "--analyze_overlap": ""
}

# ================= 3. EXPERIMENT REGISTRY =================

# --- 3.1 Core ablations (indices 1-5) ---
CORE_ABLATIONS = [
    {"name": "01_Naive_Finetuning", "args": {"--disable_dbn": "", "--ewc_lambda": "0.0", "--kd_lambda": "0.0", "--repulsive_alpha": "0.0", "--analyze_overlap": ""}},
    {"name": "02_EWC_Only", "args": {"--disable_dbn": "", "--ewc_lambda": "5000.0", "--kd_lambda": "0.0", "--repulsive_alpha": "0.0", "--analyze_overlap": ""}},
    {"name": "03_Standard_CL", "args": {"--disable_dbn": "", "--ewc_lambda": "5000.0", "--kd_lambda": "1.0", "--repulsive_alpha": "0.0", "--analyze_overlap": ""}},
    # {"name": "04_Phase1_MSBN", "args": {"--ewc_lambda": "5000.0", "--kd_lambda": "1.0", "--repulsive_alpha": "0.0", "--analyze_overlap": ""}},
    {"name": "04_Static_Repul", "args": {"--ewc_lambda": "5000.0", "--kd_lambda": "1.0", "--repulsive_alpha": "0.5", "--repulsive_margin": "0.3", "--disable_curriculum": "", "--analyze_overlap": ""}},
    {"name": "05_Ours_Full", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.3", "--p_degree": "5.0"}}
]

# --- 3.2 Gamma(p) ablations at m=0.1 (indices 6-11) ---
GAMMA_OLD_ABLATIONS = [
    {"name": f"06_p1", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.3", "--p_degree": "1.0"}},
    {"name": f"07_p3", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.3", "--p_degree": "3.0"}},
    {"name": f"08_p8",  "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.3", "--p_degree": "8.0"}},
    {"name": f"09_p.1", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.3", "--p_degree": "0.1"}},
    {"name": f"10_p.3", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.3", "--p_degree": "0.3"}},
    {"name": f"11_p.5", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.3", "--p_degree": "0.5"}},
]

# --- 3.3 Margin(m) sweep (indices 12-25) ---
M_MARGINS = [-0.2, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0]
M_ABLATIONS = []
for idx, m_val in enumerate(M_MARGINS, start=12):
    M_ABLATIONS.append({
        "name": f"{idx}_mT_{m_val}",
        "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": str(m_val), "--p_degree": "5.0"}
    })

# --- 3.4 Gamma(p) sweep at optimal m=0.7 (indices 26-32) ---
OPT_M = "0.7"
NEW_P_DEGREES = ["0.1", "0.3", "0.5", "1.0", "3.0", "5.0", "8.0"]
GAMMA_NEW_ABLATIONS = []
for idx, p_val in enumerate(NEW_P_DEGREES, start=26):
    GAMMA_NEW_ABLATIONS.append({
        "name": f"{idx}_OptM0.7_p{p_val}",
        "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": OPT_M, "--p_degree": p_val}
    })

# --- 3.5 🏆 External SOTA baselines (indices 33-35) ---
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

# ================= Merge all experiment configs =================
ALL_EXPERIMENTS = CORE_ABLATIONS + GAMMA_OLD_ABLATIONS + M_ABLATIONS + GAMMA_NEW_ABLATIONS + EXTERNAL_BASELINES

# ================= 4. EXECUTION ENGINE =================
def build_command(exp_dict, seed, gpu_id):
    # Route baselines to their script; else weargait_train2.py
    is_baseline = "script" in exp_dict
    script_name = exp_dict.get("script", "weargait_train2.py")
    script_path = os.path.join(SCRIPT_DIR, script_name)
    
    cmd = ["python", "-u", script_path]
    
    # 1. Inject global base args
    merged = COMMON_ARGS.copy()
    
    # 2. Inject method-specific args (skip for baselines)
    if not is_baseline:
        merged.update(OURS_BASE_ARGS)
        
    # 3. Inject experiment-specific args
    merged.update(exp_dict["args"])
    
    # 4. Inject device and seed
    merged["--seed"] = str(seed)
    merged["--device"] = f"cuda:{gpu_id}"
    
    # 5. Attach CSV log path for our method only
    if not is_baseline:
        csv_path = os.path.join(LOG_BASE, exp_dict["name"], f"seed_{seed}_curves.csv")
        merged["--csv_log"] = csv_path
    
    # Build CLI command
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
# 🛠️ Execution cheatsheet
# =========================================================================================
"""
Fine-grained experiment control via -r (indices) and -g (GPU pool).
Indices follow ALL_EXPERIMENTS order (1-35).

[1] External SOTA baselines (DRMN, Harmony, LwI):
nohup python -u run_ablations.py -r 33 34 35 -g 0 1 > log_baselines.out 2>&1 &

[2] Single baseline (e.g. LwI OT):
nohup python -u run_ablations.py -r 35 -g 0 1 > log_lwi_only.out 2>&1 &

[3] Core five-step ablations:
nohup python -u run_ablations.py -r 1 2 3 4 5 -g 0 1 > log_core_ablations.out 2>&1 &

[4] Margin(m) grid (indices 12-25):
nohup python -u run_ablations.py -r 12 13 14 15 16 17 18 19 20 21 22 23 24 25 -g 0 1 > log_margin_search.out 2>&1 &

[5] Gamma(p) at OptM=0.7 (indices 26-32):
nohup python -u run_ablations.py -r 26 27 28 29 30 31 32 -g 0 1 > log_gamma_new.out 2>&1 &

[6] Run all 35 configurations:
nohup python -u run_ablations.py -g 0 1 2 3 > log_all_suite.out 2>&1 &

Tips:
- Load balance across GPUs: -g 0 1 2 3 spreads seeds across cards.
- Resume after crash: re-run unfinished indices with -r; completed logs are kept.
"""