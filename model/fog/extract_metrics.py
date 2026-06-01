import os
import re
import numpy as np

# 1. Mathematical Constants & Definitions
# Upper-bound Oracle performance (\Omega_j)
ORACLES = [36.92, 42.67, 42.47] 
TASKS = ['skeleton', 'gyr', 'acc']
T = len(TASKS)

from model.paths import FOG_BASELINES_LOG, as_str

# Target log directories based on your file structure
BASE_DIR = as_str(FOG_BASELINES_LOG)
MODELS = ['DRMN', 'Harmony', 'LwI']
SEEDS = ['seed_3.out', 'seed_4.out', 'seed_42.out', 'seed_43.out', 'seed_44.out']

def parse_log_file(file_path):
    """
    Parses a single CL log file and constructs the performance matrices for all folds.
    Returns: A numpy array of shape (n_folds, T, T)
    """
    if not os.path.exists(file_path):
        print(f"[Warning] File missing: {file_path}")
        return None
        
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Isolate folds using your explicit delimiters
    folds = re.split(r'==================== Fold \d+/\d+ ====================', content)[1:]
    if not folds:
        print(f"[Warning] No fold boundaries detected in {file_path}")
        return None
        
    R_folds = []
    
    for fold_idx, fold_text in enumerate(folds):
        R = np.zeros((T, T))
        for step in range(1, T + 1):
            # Extract the evaluation block for Step t
            block_match = re.search(rf'--- Evaluation \(Step {step}\) ---\n(.*?)(?=\n\n|\n===|\Z)', fold_text, re.DOTALL)
            if not block_match:
                continue
            
            block = block_match.group(1)
            
            # Extract R_{t, j} for j <= t
            for j in range(step):
                task_name = TASKS[j]
                val_match = re.search(rf'^\s*{task_name}:\s*([\d\.]+)', block, re.MULTILINE)
                if val_match:
                    R[step-1, j] = float(val_match.group(1))
                else:
                    print(f"Error parsing task '{task_name}' at Step {step}, Fold {fold_idx+1} in {file_path}")
                    
        R_folds.append(R)
        
    return np.array(R_folds)

def calculate_cl_metrics(R_matrix):
    """
    Computes rigorous CL metrics (A_N, BWT, NAA) from a T x T performance matrix.
    """
    # Final Average Accuracy/F1 (A_N)
    A_N = np.mean(R_matrix[-1, :])
    
    # Backward Transfer (BWT)
    bwt_sum = sum(R_matrix[-1, j] - R_matrix[j, j] for j in range(T - 1))
    BWT = bwt_sum / (T - 1)
    
    # Normalized Average Accuracy (NAA)
    naa_sum = sum(R_matrix[-1, j] / ORACLES[j] for j in range(T))
    NAA = (naa_sum / T) * 100
    
    return A_N, BWT, NAA

def main():
    print("="*60)
    print("CONTINUAL LEARNING METRICS EXTRACTION".center(60))
    print("="*60)
    
    for model in MODELS:
        print(f"\n>>> Analyzing Model: {model}")
        seed_metrics = []
        
        for seed_file in SEEDS:
            file_path = os.path.join(BASE_DIR, model, seed_file)
            R_folds = parse_log_file(file_path)
            
            if R_folds is not None and len(R_folds) > 0:
                # Average across folds to get the stabilized matrix for this seed
                R_seed_avg = np.mean(R_folds, axis=0)
                a_n, bwt, naa = calculate_cl_metrics(R_seed_avg)
                seed_metrics.append((a_n, bwt, naa))
                print(f"  [{seed_file}] A_N: {a_n:.2f}% | BWT: {bwt:.2f}% | NAA: {naa:.2f}%")
                
        if seed_metrics:
            seed_metrics = np.array(seed_metrics)
            avg_a_n, std_a_n = np.mean(seed_metrics[:, 0]), np.std(seed_metrics[:, 0])
            avg_bwt, std_bwt = np.mean(seed_metrics[:, 1]), np.std(seed_metrics[:, 1])
            avg_naa, std_naa = np.mean(seed_metrics[:, 2]), np.std(seed_metrics[:, 2])
            
            print("-" * 60)
            print(f"FINAL STATISTICS (Across {len(seed_metrics)} Seeds) - {model}")
            print(f"A_N : {avg_a_n:.2f} +/- {std_a_n:.2f}%")
            print(f"BWT : {avg_bwt:.2f} +/- {std_bwt:.2f}%")
            print(f"NAA : {avg_naa:.2f} +/- {std_naa:.2f}%")
            print("="*60)

if __name__ == "__main__":
    main()