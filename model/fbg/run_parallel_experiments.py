import os
import subprocess
import time

from model.paths import FBG_PROCESSED, as_str

def main():
    # ==========================================
    # 1. 配置：指向新引擎 fbg_cl_train.py
    # ==========================================
    DATA_ROOT = as_str(FBG_PROCESSED)
    MODALITIES = ["linear", "angular", "grf"]
    SEEDS = [2, 3, 42, 43, 44]
    
    AVAILABLE_GPUS = [0, 1] 
    LOG_DIR = "./experiment_logs/specialist_baselines"
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print("=" * 60)
    print(f" 🚀 拉起 Specialist Oracle 基线流水线")
    print("=" * 60)

    task_counter = 0
    
    for mod in MODALITIES:
        for seed in SEEDS:
            gpu_id = AVAILABLE_GPUS[task_counter % len(AVAILABLE_GPUS)]
            task_counter += 1
            
            log_file = os.path.join(LOG_DIR, f"oracle_{mod}_seed_{seed}.log")
            
            cmd = (
                f"python -u fbg_train.py "
                f"--data_root {DATA_ROOT} "
                f"--order {mod} "
                f"--seed {seed} "
                f"> {log_file} 2>&1"
            )
            
            current_env = os.environ.copy()
            current_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            
            print(f"[*] 任务 {task_counter:02d}/15 | 模态: {mod:<7} | 种子: {seed:<4} -> GPU: {gpu_id}")
            
            # 使用 Popen 执行，确保不阻塞
            subprocess.Popen(cmd, shell=True, env=current_env)
            time.sleep(0.5)

    print("=" * 60)
    print(f" 15 个 Specialist 基线进程已拉起。")
    print("=" * 60)

if __name__ == "__main__":
    main()