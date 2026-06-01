import os
import stat

# ================= CONFIGURATION =================
# Where your logs are stored
LOG_DIR = "/home/yongjie/Minlin/IJCAI_26/model/weargait/LwI/log"

# IMPORTANT: The root folder where python -m should run
PROJECT_ROOT = "/home/yongjie/Minlin" 

# Grid Search Parameters
SEEDS = [2, 3, 4, 42, 43, 44]
STEPS = [0.6, 0.7, 0.8, 0.9, 1.0]
LAMBDAS = [0.5, 0.7, 0.9, 1.0]

SUCCESS_MARKER = "--- Evaluation after Task 2"
RERUN_SCRIPT_NAME = "rerun_missing_jobs.sh"

# ================= LOGIC =================

def generate_rerun_script():
    missing_count = 0
    commands = []

    print(f"Scanning {LOG_DIR} for missing or crashed runs...")

    for s in SEEDS:
        for st in STEPS:
            for ld in LAMBDAS:
                filename = f"ot_s{s}_step{st}_ld{ld}.out"
                filepath = os.path.join(LOG_DIR, filename)
                
                is_missing = False
                reason = ""

                if not os.path.exists(filepath):
                    is_missing = True
                    reason = "File not found"
                else:
                    try:
                        with open(filepath, 'r') as f:
                            content = f.read()
                            if SUCCESS_MARKER not in content:
                                is_missing = True
                                reason = "Incomplete/Crashed"
                    except Exception as e:
                        is_missing = True
                        reason = f"Read Error: {e}"

                if is_missing:
                    print(f"  [MISSING] Seed={s}, Step={st}, Ld={ld} ({reason})")
                    
                    cmd = (
                        f"nohup python -u -m IJCAI_26.model.weargait.LwI.ot_train "
                        f"--seed {s} --step {st} --kd_lambda {ld} "
                        f"> {filepath} 2>&1 &"
                    )
                    commands.append(cmd)
                    missing_count += 1

    # ================= OUTPUT =================
    if missing_count == 0:
        print("\nSUCCESS: All configurations have run successfully!")
        if os.path.exists(RERUN_SCRIPT_NAME):
            os.remove(RERUN_SCRIPT_NAME)
    else:
        print(f"\nFound {missing_count} missing/failed runs.")
        print(f"Generating {RERUN_SCRIPT_NAME}...")
        
        with open(RERUN_SCRIPT_NAME, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Auto-generated rerun script\n")
            
            # --- FIX: Force the script to go to the project root first ---
            f.write(f"cd {PROJECT_ROOT}\n\n") 
            
            for cmd in commands:
                f.write(cmd + "\n")
                f.write("sleep 0.2\n")
        
        st = os.stat(RERUN_SCRIPT_NAME)
        os.chmod(RERUN_SCRIPT_NAME, st.st_mode | stat.S_IEXEC)
        
        print(f"Done. Run './{RERUN_SCRIPT_NAME}' to execute them.")

if __name__ == "__main__":
    generate_rerun_script()

# ./rerun_missing_jobs.sh