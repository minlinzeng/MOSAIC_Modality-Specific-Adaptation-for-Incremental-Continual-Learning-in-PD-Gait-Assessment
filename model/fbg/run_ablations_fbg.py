import os
import subprocess
import time
import argparse
import itertools
from datetime import datetime
import itertools

from model.paths import FBG_PROCESSED, as_str

def generate_ablation_matrix():
    ABLATIONS = {}
    BEST_EWC = "5000.0"
    BEST_KD = "1.0"
    BEST_ALPHA = "0.5"
    
    # =====================================================
    # [Task 02 series]: Curriculum & Margin dynamics tuning (3x3=9 runs)
    # Run: python run_ablations_fbg.py -r 02 -g "0,1"
    # =====================================================
    margin_list = [0.1, 0.3, 0.5]         
    p_degree_list = [3.0, 5.0, 7.0]       
    
    idx = 1
    for margin, p_deg in itertools.product(margin_list, p_degree_list):
        exp_name = f"02_Dynamics_{idx:02d}_Margin{margin}_Pdeg{p_deg}"
        ABLATIONS[exp_name] = [
            "--lambda_ewc", BEST_EWC,
            "--lambda_kd", BEST_KD,
            "--alpha_max", BEST_ALPHA,
            "--repulsive_margin", str(margin),
            "--p_degree", str(p_deg)
        ]
        idx += 1

    # =====================================================
    # [Task 03~07 series]: Baseline comparison and ablations (numeric prefix only)
    # Run: python run_ablations_fbg.py -r "03,04,05,06,07" -g "0,1"
    # =====================================================
    
    # 03. Vanilla finetune baseline (no MSBN, no constraints)
    ABLATIONS["03_Vanilla_Finetune"] = [
        "--lambda_ewc", "0.0", "--lambda_kd", "0.0", "--alpha_max", "0.0", 
        "--disable_curriculum", "--disable_msbn"
    ]
    
    # 04. Vanilla EWC baseline (parameter-space anti-forgetting only, no MSBN)
    ABLATIONS["04_Vanilla_EWC"] = [
        "--lambda_ewc", BEST_EWC, "--lambda_kd", "0.0", "--alpha_max", "0.0", 
        "--disable_curriculum", "--disable_msbn"
    ]
    
    # 05. Vanilla EWC+LwF strong baseline (parameter + activation anti-forgetting, no MSBN)
    ABLATIONS["05_Vanilla_EWC_LwF"] = [
        "--lambda_ewc", BEST_EWC, "--lambda_kd", BEST_KD, "--alpha_max", "0.0", 
        "--disable_curriculum", "--disable_msbn"
    ]
    
    # 06. Ours static variant (MSBN + static repulsive loss, curriculum disabled)
    # Demonstrates necessity of polynomial curriculum scheduling
    ABLATIONS["06_Ours_Static"] = [
        "--lambda_ewc", BEST_EWC, "--lambda_kd", BEST_KD, "--alpha_max", BEST_ALPHA, 
        "--disable_curriculum"
    ]
    
    # 07. Ours full model (all optimal design choices enabled)
    ABLATIONS["07_Ours_Full"] = [
        "--lambda_ewc", BEST_EWC, "--lambda_kd", BEST_KD, "--alpha_max", BEST_ALPHA
    ]
    
    return ABLATIONS

ABLATIONS = generate_ablation_matrix()

# =====================================================================
# 2. High-concurrency round-robin dispatcher
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="FBG CL Ablation & Grid Search Dispatcher")
    parser.add_argument('-g', '--gpu', type=str, default="0,1",
                        help="GPU IDs to use (e.g. '0,1')")
    parser.add_argument('-r', '--runs', type=str, default="all",
                        help="Experiment ID prefix to run (e.g. '01' for tuning, 'all' for all)")
    parser.add_argument('-j', '--jobs_per_gpu', type=int, default=10,
                        help="Max concurrent jobs per physical GPU")
    parser.add_argument('--data_root', type=str, default=as_str(FBG_PROCESSED))
    return parser.parse_args()

def main():
    args = parse_args()
    
    script_to_run = "fbg_cl_train.py"
    task_order = "linear,angular,grf"
    log_dir = "logs_fbg_ablations"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Route / filter requested runs
    target_runs = {}
    if args.runs.lower() == "all":
        target_runs = ABLATIONS
    else:
        requested_ids = [rid.strip() for rid in args.runs.split(",")]
        for exp_name, cmd_args in ABLATIONS.items():
            exp_id = exp_name.split("_")[0]
            if exp_id in requested_ids or str(int(exp_id)) in requested_ids:
                target_runs[exp_name] = cmd_args
                
    if not target_runs:
        print("[!] Error: no experiments matched the requested run IDs.")
        return

    # Strict round-robin GPU allocation
    physical_gpus = [g.strip() for g in args.gpu.split(",")]
    total_slots = args.jobs_per_gpu * len(physical_gpus)
    logical_gpu_slots = []
    
    for i in range(total_slots):
        logical_gpu_slots.append(physical_gpus[i % len(physical_gpus)])

    print(f"\n{'='*70}")
    print(f" 🚀 FBG GRID SEARCH & ABLATION DISPATCHER")
    print(f" Physical GPUs : {physical_gpus}")
    print(f" Jobs per GPU  : {args.jobs_per_gpu}")
    print(f" Total Capacity: {total_slots} Concurrent Processes")
    print(f" Allocation Map: {logical_gpu_slots}") 
    print(f" Planned Runs  : {len(target_runs)} Experiments queued.")
    print(f"{'='*70}\n")
    
    if not os.path.exists(script_to_run):
        raise FileNotFoundError(f"Fatal error: cannot find {script_to_run}")

    active_processes = []  
    job_queue = list(target_runs.items())
    completed_jobs = 0
    total_jobs = len(job_queue)

    while job_queue or active_processes:
        # Reclaim finished processes
        for p_info in active_processes[:]:
            process, gpu_id, exp_name, log_file, start_time = p_info
            
            if process.poll() is not None:
                elapsed = (time.time() - start_time) / 60
                log_file.close()
                active_processes.remove(p_info)
                logical_gpu_slots.append(gpu_id) 
                completed_jobs += 1
                
                if process.returncode == 0:
                    print(f"      [OK] {exp_name} finished (GPU: {gpu_id}), elapsed: {elapsed:.1f} min. [{completed_jobs}/{total_jobs}]")
                else:
                    print(f"      [!] {exp_name} failed (GPU: {gpu_id})! Check logs. [{completed_jobs}/{total_jobs}]")

        # Job launcher
        while logical_gpu_slots and job_queue:
            exp_name, specific_args = job_queue.pop(0)
            gpu_id = logical_gpu_slots.pop(0) 
            
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
            
            base_cmd = ["python", "-u", script_to_run, "--data_root", args.data_root, "--order", task_order]
            full_cmd = base_cmd + specific_args
            
            log_filename = f"grid_{exp_name}_{timestamp}.log"
            log_filepath = os.path.join(log_dir, log_filename)
            
            print(f"      [>>>] Launch: {exp_name} --> GPU: {gpu_id}")
            
            # Log file setup
            log_file = open(log_filepath, "w")
            log_file.write(f"=== EXPERIMENT: {exp_name} ===\n")
            log_file.write(f"CUDA_VISIBLE_DEVICES: {gpu_id}\n")
            log_file.write(f"Command: {' '.join(full_cmd)}\n")
            log_file.write(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write("-" * 50 + "\n\n")
            
            process = subprocess.Popen(
                full_cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env, text=True
            )
            
            active_processes.append((process, gpu_id, exp_name, log_file, time.time()))
            
            time.sleep(3.0) 
            
        time.sleep(1)

    print(f"\n{'='*70}")
    print(f" All grid experiments finished.")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()