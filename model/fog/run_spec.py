import os
import time
import subprocess
from queue import Queue
from threading import Thread

# =====================================================================
# 🎛️ Physics constraints and hyperparameter grid
# =====================================================================
SCRIPT_NAME = "fog_train.py"
LOG_DIR = "./log/spec"
CSV_DIR = "./log/spec/csv"
AVAILABLE_GPUS = [0, 1]            # Target GPU IDs
MAX_PER_GPU = 15                    # Max concurrent jobs per GPU

SEEDS = [3, 4, 42, 43, 44]
MODALITIES = ["acc", "gyr", "skeleton"]

# =====================================================================
# ⚙️ Scheduler (producer-consumer)
# =====================================================================
def worker(gpu_id, task_queue):
    while not task_queue.empty():
        try:
            task = task_queue.get_nowait()
        except Exception:
            break
            
        seed = task['seed']
        mod = task['mod']
        
        os.makedirs(LOG_DIR, exist_ok=True)
        os.makedirs(CSV_DIR, exist_ok=True)
        
        log_file = os.path.join(LOG_DIR, f"specialist_{mod}_seed{seed}.log")
        csv_path = os.path.join(CSV_DIR, f"specialist_{mod}_seed{seed}.csv")
        
        # 🚨 Map to fog_train.py CLI args
        cmd = [
            "python", "-u", SCRIPT_NAME,
            "--seed", str(seed),
            "--order", mod,              # Single-modality oracle (no CL stream)
            "--disable_dbn",             # Disable DBN for single distribution
            "--epochs", "50",           # Override default 80 epochs
            "--batch_size", "32",        # Default batch size to avoid OOM
            "--num_workers", "2",        # Limit DataLoader workers
            "--csv_log", csv_path        # Per-run CSV log path
        ]
        
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        
        print(f"🚀 [GPU {gpu_id}] 引擎点火 | 模态: {mod.upper():<8} | Seed: {seed} | 日志: {os.path.basename(log_file)}")
        start_time = time.time()
        
        with open(log_file, "w") as f:
            process = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
            process.wait()
            
        duration = (time.time() - start_time) / 60.0
        
        if process.returncode == 0:
            print(f"✅ [GPU {gpu_id}] 任务完成 | 模态: {mod.upper():<8} | Seed: {seed} | 耗时: {duration:.1f} min")
        else:
            print(f"❌ [GPU {gpu_id}] 任务崩溃 | 模态: {mod.upper():<8} | Seed: {seed} | Exit Code: {process.returncode}")
            
        task_queue.task_done()

def main():
    task_queue = Queue()
    total_tasks = 0
    for mod in MODALITIES:
        for seed in SEEDS:
            task_queue.put({'seed': seed, 'mod': mod})
            total_tasks += 1
            
    print(f"\n{'='*60}\n 🎯 EXPERIMENT SCHEDULER INITIALIZED \n{'='*60}")
    print(f"  总任务数 (N)       : {total_tasks}")
    print(f"  算力矩阵 (GPUs)    : {AVAILABLE_GPUS}")
    print(f"  系统并发度 (C)     : {len(AVAILABLE_GPUS) * MAX_PER_GPU}\n")

    threads = []
    for gpu_id in AVAILABLE_GPUS:
        for _ in range(MAX_PER_GPU):
            t = Thread(target=worker, args=(gpu_id, task_queue))
            t.start()
            threads.append(t)
            
    for t in threads:
        t.join()
        
    print(f"\n{'='*60}\n 🏆 ALL EXPERIMENTS COMPLETED \n{'='*60}\n")

if __name__ == "__main__":
    main()