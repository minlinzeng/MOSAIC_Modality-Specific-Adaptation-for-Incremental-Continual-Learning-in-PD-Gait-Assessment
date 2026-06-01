import os
import subprocess
import time
import argparse

# ================= 1. Environment and hardware =================
SCRIPT_PATH = "fog_train.py" 
LOG_BASE = "./log/ablations/fog"
SEEDS = [42, 43, 44, 2, 3]  

# ================= 2. FOG default hyperparameters (STATIC) =================
# FOG small-sample / noisy setting
BASE_ARGS = {
    "--mode": "cl",
    "--order": "skeleton,gyr,acc",
    "--batch_size": "32",   # Small batch for regularization
    "--lr": "0.0001",       # LR scaled with batch size
    "--lr_we": "10",
    "--epochs": "80",       
    "--patience": "20",            
    "--win_len": "120",
    "--hop_len": "15",      # Dense hop_len for more windows
    "--num_workers": "4",
    "--num_classes": "3",   # FOG H&Y 3-class
    "--kd_we": "10",
    "--fisher_batches": "16",
}

# ================= 3. FOG ablation matrix =================
# Shared repulsive args
COMMON_REPUL_ARGS = {
    "--ewc_lambda": "5000.0", 
    "--kd_lambda": "1.0", 
    "--min_kd_lambda": "0.1", 
    "--repulsive_alpha": "0.5", 
    "--analyze_overlap": ""
}

# --- 3.1 Core ablations ---
CORE_ABLATIONS = [
    # 0. Catastrophic forgetting lower bound (no protection)
    {"name": "00_Naive_Finetuning", "args": {"--disable_dbn": "", "--ewc_lambda": "0.0", "--kd_lambda": "0.0", "--repulsive_alpha": "0.0", "--analyze_overlap": ""}},
    
    # 1. EWC only (no KD, DBN off)
    {"name": "01_Pure_EWC",         "args": {"--disable_dbn": "", "--ewc_lambda": "5000.0", "--kd_lambda": "0.0", "--repulsive_alpha": "0.0", "--analyze_overlap": ""}},
    
    # 2. EWC + KD (LwF, DBN off)
    {"name": "02_LwF",              "args": {"--disable_dbn": "", "--ewc_lambda": "5000.0", "--kd_lambda": "1.0", "--repulsive_alpha": "0.0", "--analyze_overlap": ""}},
    
    # 3. Static repulsive (no curriculum, DBN off)
    {"name": "03_Static_Repulsive", "args": {"--disable_dbn": "", "--ewc_lambda": "5000.0", "--kd_lambda": "1.0", "--repulsive_alpha": "0.5", "--repulsive_margin": "0.1", "--disable_curriculum": "", "--analyze_overlap": ""}},
    
    # 4. Full method (DBN + repulsive + curriculum)
    {"name": "04_Ours_Full",        "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.1", "--p_degree": "5.0"}}
]

# --- 3.2 Gamma(p) ablation at m=0.1 (indices 6-11) ---
# For paper gamma curves
GAMMA_ABLATIONS = [
    {"name": f"06_p0.1", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.1", "--p_degree": "0.1"}},
    {"name": f"07_p0.3", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.1", "--p_degree": "0.3"}},
    {"name": f"08_p0.5", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.1", "--p_degree": "0.5"}},
    {"name": f"09_p1.0", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.1", "--p_degree": "1.0"}},
    {"name": f"10_p3.0", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.1", "--p_degree": "3.0"}},
    {"name": f"11_p8.0", "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": "0.1", "--p_degree": "8.0"}},
]

# --- 3.3 Margin(m) ablation (indices 12-16) ---
# Margins capped at 0.4 (skip unstable 0.7 region)
M_MARGINS = [-0.2, 0.0, 0.2, 0.3, 0.4] # 0.1 covered in 05_Ours_Full
M_ABLATIONS = []
for idx, m_val in enumerate(M_MARGINS, start=12):
    M_ABLATIONS.append({
        "name": f"{idx}_mT_{m_val}",
        "args": {**COMMON_REPUL_ARGS, "--repulsive_margin": str(m_val), "--p_degree": "5.0"}
    })

# --- 3.4 Misc test (index 17) ---
MISC_ABLATIONS = [
    {
        "name": "17_Standard_EWC_KD_we30", 
        "args": {
            "--disable_dbn": "",          
            "--ewc_lambda": "5000.0",     
            "--kd_lambda": "1.0",         
            "--repulsive_alpha": "0.0",
            "--analyze_overlap": "",
            "--kd_we": "30" # Longer warmup ablation
        }
    }
]

# ================= Merge all experiment configs =================
ABLATIONS = CORE_ABLATIONS + GAMMA_ABLATIONS + M_ABLATIONS + MISC_ABLATIONS

# ================= 4. Execution engine =================
def build_command(exp_args, seed, gpu_id, exp_name):
    cmd = ["python", "-u", SCRIPT_PATH]
    
    merged = BASE_ARGS.copy()
    merged.update(exp_args)
    
    merged["--seed"] = str(seed)
    merged["--device"] = f"cuda:{gpu_id}"
    
    csv_path = os.path.join(LOG_BASE, exp_name, f"seed_{seed}_curves.csv")
    merged["--csv_log"] = csv_path
    
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
        experiments = [ABLATIONS[i-1] for i in target_runs if 0 < i <= len(ABLATIONS)]
        if not experiments:
            print("❌ Invalid experiment number(s) provided. Check your input.")
            return
    else:
        experiments = ABLATIONS

    current_max_parallel = len(gpus) * 15 

    print(f"🚀 Starting FOG Ablation Suite: {len(experiments)} configurations x {len(SEEDS)} seeds.")
    print(f"🖥️  Targeting GPUs: {gpus} (Max {current_max_parallel} concurrent jobs)")
    
    for ablation in experiments:
        exp_name = ablation["name"]
        exp_dir = os.path.join(LOG_BASE, exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        
        print("\n" + "="*50)
        print(f"   Initiating: {exp_name}")
        print("="*50)
        
        for seed in SEEDS:
            while len(active_processes) >= current_max_parallel:
                active_processes = [p for p in active_processes if p.poll() is None]
                time.sleep(2) 
            
            gpu_id = gpus[global_counter % len(gpus)]
            cmd = build_command(ablation["args"], seed, gpu_id, exp_name)
            log_file = os.path.join(exp_dir, f"seed_{seed}.out")
            
            print(f"  [{exp_name} | Seed {seed}] Queueing on GPU {gpu_id}...")
            
            with open(log_file, "w") as f:
                p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
                active_processes.append(p)
            
            global_counter += 1
            time.sleep(1.0) 

    print("\n⏳ All selected jobs submitted! Waiting for final batch to complete...")
    for p in active_processes:
        p.wait()
    print("✅ Selected FOG Ablation Suite Completely Finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FOG MICL Ablation Orchestrator")
    
    parser.add_argument('-r', '--run', nargs='+', type=int, 
                        help='Specify which experiments to run (e.g., -r 4  or  --run 4 5). Runs all if omitted.')
    
    parser.add_argument('-g', '--gpus', nargs='+', type=int, default=[0, 1], 
                        help='Specify GPU IDs to use (e.g., -g 1  or  -g 0 2). Default: 0 1')
    
    args = parser.parse_args()
    
    run_suite(args.run, args.gpus)