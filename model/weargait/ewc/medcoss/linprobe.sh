#!/bin/bash

# ==========================================
# MedCoSS CL Evaluation: Linear Probing
# ==========================================

BASE_DIR="/home/minlin/Minlin/JBHI26/model/weargait/ewc/medcoss"
LOG_DIR="${BASE_DIR}/log/linprobe_results"

# Create a clean directory for all the F1 matrix logs
mkdir -p ${LOG_DIR}

# Array of 6 distinct random seeds for stability testing
SEEDS=(2 3 4 42 43 44)

# Define exact weights for each CL step
# (Update CKPT_STEP1 to match your original Stage 1 IMU weights)
CKPT_STEP1="${BASE_DIR}/log/checkpoint-199.pth" 
CKPT_STEP2="${BASE_DIR}/log/stage3_walkway/checkpoint-199.pth"
CKPT_STEP3="${BASE_DIR}/log/stage3_insole/checkpoint-199.pth"

echo "🚀 Initiating 6-Seed Linear Probing Evaluation..."

for SEED in "${SEEDS[@]}"; do
    echo "========================================"
    echo " 🎯 RUNNING SEED: ${SEED}"
    echo "========================================"

    # ---------------------------------------------------------
    # ROW 1: Evaluate Step 1 (IMU)
    # ---------------------------------------------------------
    # echo "Evaluating Row 1 (Step 1 weights)..."
    # python -u ${BASE_DIR}/main_linprobe.py \
    #     --eval_step 1 \
    #     --seen_tasks "1D_text" \
    #     --load_pretrained_weight ${CKPT_STEP1} \
    #     --seed ${SEED} \
    #     --output_dir ${LOG_DIR}/step1_seed${SEED} \
    #     > ${LOG_DIR}/eval_step1_seed${SEED}.out 2>&1

    # ---------------------------------------------------------
    # ROW 2: Evaluate Step 2 (IMU + Walkway)
    # ---------------------------------------------------------
    # echo "Evaluating Row 2 (Step 2 weights)..."
    # python -u ${BASE_DIR}/main_linprobe.py \
    #     --eval_step 2 \
    #     --seen_tasks "1D_text,2D_xray" \
    #     --load_pretrained_weight ${CKPT_STEP2} \
    #     --seed ${SEED} \
    #     --output_dir ${LOG_DIR}/step2_seed${SEED} \
    #     > ${LOG_DIR}/eval_step2_seed${SEED}.out 2>&1

    # ---------------------------------------------------------
    # ROW 3: Evaluate Step 3 (IMU + Walkway + Insole)
    # ---------------------------------------------------------
    echo "Evaluating Row 3 (Step 3 weights)..."
    python -u ${BASE_DIR}/main_linprobe.py \
        --eval_step 3 \
        --seen_tasks "1D_text,2D_xray,2D_path" \
        --load_pretrained_weight ${CKPT_STEP3} \
        --seed ${SEED} \
        --output_dir ${LOG_DIR}/step3_seed${SEED} \
        > ${LOG_DIR}/eval_step3_seed${SEED}.out 2>&1

done

echo "✅ All 18 evaluation runs completed. Logs saved to: ${LOG_DIR}"