import os
import time
import subprocess
from queue import Queue
from threading import Thread

# =====================================================================
# 🎛️ 物理约束与实验超参数矩阵
# =====================================================================
SCRIPT_NAME = "fog_train.py"
LOG_DIR = "./log/spec"
CSV_DIR = "./log/spec/csv"
AVAILABLE_GPUS = [0, 1]            # 挂载的目标物理 GPU
MAX_PER_GPU = 15                    # 单卡算力锁死，严禁显存踩踏

SEEDS = [3, 4, 42, 43, 44]
MODALITIES = ["acc", "gyr", "skeleton"]

# =====================================================================
# ⚙️ 核心调度引擎 (Producer-Consumer Topology)
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
        
        # 🚨 核心映射：严格适配 fog_train.py 的超参数签名
        cmd = [
            "python", "-u", SCRIPT_NAME,
            "--seed", str(seed),
            "--order", mod,              # 强制降维：仅投喂单一物理源，切断 CL 任务流
            "--disable_dbn",             # 剥离多任务域归一化，锁定单一分布参数
            "--epochs", "50",           # 覆盖默认 80 轮限制，探明极限收敛点
            "--batch_size", "32",        # 对齐默认张量体积，防止 OOM
            "--num_workers", "2",        # 压制 CPU 调度器线程暴增
            "--csv_log", csv_path        # 挂载独立数据遥测日志
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