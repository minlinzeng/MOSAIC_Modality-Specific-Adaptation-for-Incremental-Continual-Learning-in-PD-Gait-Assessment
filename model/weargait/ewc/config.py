from pathlib import Path

from model.paths import (
    WEARGAIT_CHECKPOINTS,
    WEARGAIT_DATA,
    WEARGAIT_HC_RAW,
    WEARGAIT_PD_RAW,
)

class Config:
    # ================= 1. FILE PATHS =================
    HC_PATH = WEARGAIT_HC_RAW
    PD_PATH = WEARGAIT_PD_RAW
    OUTPUT_DIR = WEARGAIT_DATA
    HC_DEMO_CSV = HC_PATH / "hc_demographic.csv"
    PD_DEMO_CSV = PD_PATH / "pd_demographic.csv"
    CHECKPOINT_DIR = WEARGAIT_CHECKPOINTS

    # ================= 2. PHYSICS CONSTANTS =================
    GRAV = 9.81
    TARGET_HZ = 30 # downsample to this hz
    CSV_PATTERN = "*_SelfPace_matTURN.csv"
    EXCLUDED_EVENTS = ["standing", "unlabeled"] 

    # ================= 3. SENSOR DEFINITIONS (THE HEADER BUILDERS) =================
    # A. IMU SITES (13 Body Locations + 2 Insoles)
    # Source: "13 sensors on the participant's body" [cite: 171] + 2 insoles [cite: 172]
    BODY_IMU_SITES = [
        "LowerBack", "Xiphoid", "Forehead",
        "R_Wrist", "L_Wrist",
        "R_MidLatThigh", "L_MidLatThigh",
        "R_LatShank", "L_LatShank",
        "R_Ankle", "L_Ankle",
        "R_DorsalFoot", "L_DorsalFoot",
        "Rinsole", "Linsole"
    ]
    
    IMU_MODALITIES = {
        "Acc":  ["FreeAcc_E", "FreeAcc_N", "FreeAcc_U"], # Gravity Removed
        "Gyr":  ["Gyr_X", "Gyr_Y", "Gyr_Z"],             # Rotational Velocity
        # "Mag": ["Mag_X", "Mag_Y", "Mag_Z"]             # Optional: Uncomment if you want Mag later
    }

    # C. INSOLE DEFINITIONS
    # Source: "16 pressure sensors in each insole" [cite: 172]
    INSOLE_SENSORS_COUNT = 16 
    INSOLE_COLS = [
        "LTotalForce_BW", "RTotalForce_BW", 
        "LCoP_X", "LCoP_Y", "RCoP_X", "RCoP_Y"
    ]

    # D. WALKWAY DEFINITIONS
    WALKWAY_COLS = [
        "L_Pressure_BW", "R_Pressure_BW", 
        "L_Contact", "R_Contact", 
        "Walkway_X_Norm", "Walkway_Vel",
        "Pressure_Diff", "Total_Load"
    ]

    # ================= 4. WINDOWING & TRAINING =================
    WINDOW_SEC = 4.0
    WINDOW_SIZE = int(WINDOW_SEC * TARGET_HZ)  # 120 frames
    STRIDE = 0.5
    
    BATCH_SIZE = 128
    LEARNING_RATE = 1e-4
    EPOCHS = 50
    SEED = 42
    N_FOLDS = 5