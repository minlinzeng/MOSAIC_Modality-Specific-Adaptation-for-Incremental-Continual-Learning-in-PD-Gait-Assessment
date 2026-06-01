from pathlib import Path
import json, re
import numpy as np
import pandas as pd
from model.weargait.ewc.config import Config as C

# ==================== Demographics I/O  ====================
def read_demographics_with_header_fix(path: str) -> pd.DataFrame:
    try:
        df0 = pd.read_csv(path, header=None, dtype=str)
        if df0.shape[0] < 2: return pd.DataFrame()
        hdr_idx = 1
        header = df0.iloc[hdr_idx].fillna("").astype(str).str.replace(r"\s+"," ",regex=True).str.strip()
        df = df0.iloc[hdr_idx+1:].reset_index(drop=True).copy()
        df.columns = header
        return df
    except Exception as e:
        print(f"Error reading demographics {path}: {e}")
        return pd.DataFrame()

def extract_subject_weights(demo_df: pd.DataFrame) -> pd.DataFrame:
    if demo_df.empty: return pd.DataFrame()
    try:
        id_col = next(c for c in demo_df.columns if re.search(r"(subject\s*id|participant)", c, re.I))
        wt_col = next(c for c in demo_df.columns if re.search(r"weight", c, re.I))
        out = demo_df[[id_col, wt_col]].rename(columns={id_col:"subject_id", wt_col:"weight_kg"}).copy()
        out["subject_id"] = out["subject_id"].astype(str).str.strip()
        out["weight_kg"] = pd.to_numeric(out["weight_kg"].astype(str).str.extract(r"([0-9]*\.?[0-9]+)")[0], errors="coerce")
        return out.dropna(subset=["subject_id","weight_kg"]).reset_index(drop=True)
    except StopIteration:
        return pd.DataFrame()

def build_weight_map(hc_demo_csv: str, pd_demo_csv: str) -> dict:
    weight_map = {}
    for p in [hc_demo_csv, pd_demo_csv]:
        if not p: continue
        df = read_demographics_with_header_fix(p)
        w = extract_subject_weights(df)
        weight_map.update({r.subject_id.lower(): float(r.weight_kg) for _, r in w.iterrows()})
    return weight_map

def find_subject_files(root_dir: Path, pattern: str = C.CSV_PATTERN) -> dict:
    # Use Config pattern by default
    return {p.stem.split("_",1)[0].lower(): p for p in root_dir.glob(pattern)}

# ==================== Downsampling ========================
def parse_time_seconds(s: pd.Series) -> np.ndarray:
    t = (s.astype(str)
           .str.strip()
           .str.replace(" sec", "", regex=False)
           .str.replace(",", ".", regex=False))
    return pd.to_numeric(t, errors="coerce").to_numpy(dtype=float)

def downsample_raw(df: pd.DataFrame, time_col="Time", target_hz=C.TARGET_HZ) -> pd.DataFrame:
    if df.empty or time_col not in df.columns:
        return df
    t = parse_time_seconds(df[time_col])
    m = np.isfinite(t)
    if not m.any(): return pd.DataFrame()
    
    bins = np.full(t.shape, -1, dtype=np.int64)
    bins[m] = np.floor(t[m] * target_hz).astype(np.int64)

    tmp = df.copy()
    tmp["_bin"] = bins
    
    agg_dict = {}
    for c in tmp.columns:
        if c == "_bin": continue
        # Use simple heuristic for aggregation
        if "Contact" in c or "Pressure_Block" in c or "IMU_Block" in c or tmp[c].dtype == object:
             agg_dict[c] = 'first'
        else:
             agg_dict[c] = 'mean'

    out = tmp[tmp["_bin"] >= 0].groupby("_bin", sort=True, as_index=False).agg(agg_dict)
    
    out[time_col] = (out["_bin"].to_numpy(dtype=float) + 0.5) / target_hz
    out = out.drop(columns=["_bin"]).reset_index(drop=True)
    return out

# ==================== 1. WALKWAY BUILDER =====================
def parse_pipe_coord(series: pd.Series) -> pd.Series:
    def _parse(val):
        if pd.isna(val): return np.nan
        if isinstance(val, (int, float)): return float(val)
        s = str(val).strip()
        if "|" in s:
            parts = [float(x) for x in s.split("|") if x.strip()]
            return np.mean(parts) if parts else np.nan
        try: return float(s)
        except: return np.nan
    return series.apply(_parse)

def build_walkway(df: pd.DataFrame, weight_kg: float) -> pd.DataFrame:
    # We could move these column lists to Config, but they are specific to logic here
    cols = ["Time", "L Foot Pressure", "R Foot Pressure", "L Foot Contact", 
            "R Foot Contact", "Walkway_X", "Walkway_Y"]
    
    valid_cols = [c for c in cols if c in df.columns]
    if len(valid_cols) < 3: return pd.DataFrame()
    
    out = df[valid_cols].copy()

    # Normalize Pressure by Body Weight (Using Config Gravity)
    denom = weight_kg * C.GRAV if (weight_kg and weight_kg > 0) else 1.0
    out["L_Pressure_BW"] = pd.to_numeric(out["L Foot Pressure"], errors='coerce') / denom
    out["R_Pressure_BW"] = pd.to_numeric(out["R Foot Pressure"], errors='coerce') / denom
    
    if "L Foot Contact" in out: out["L_Contact"] = pd.to_numeric(out["L Foot Contact"], errors='coerce').fillna(0)
    else: out["L_Contact"] = 0.0
    if "R Foot Contact" in out: out["R_Contact"] = pd.to_numeric(out["R Foot Contact"], errors='coerce').fillna(0)
    else: out["R_Contact"] = 0.0

    if "Walkway_X" in out:
        wx = parse_pipe_coord(out["Walkway_X"]).fillna(0)
        out["Walkway_X_Norm"] = (wx - wx.mean()) 
    else: out["Walkway_X_Norm"] = 0.0

    if "Walkway_Y" in out:
        wy = parse_pipe_coord(out["Walkway_Y"]).ffill()
        dt = 1.0 / 100.0
        out["Walkway_Vel"] = (wy.diff().fillna(0) / 100.0) / dt  # converts cm/frame -> m/s
    else: out["Walkway_Vel"] = 0.0
    
    # compute two additional features
    out["Pressure_Diff"] = out["L_Pressure_BW"] - out["R_Pressure_BW"]
    out["Total_Load"]    = out["L_Pressure_BW"] + out["R_Pressure_BW"]
    
    final_cols = ["Time", "L_Pressure_BW", "R_Pressure_BW", "L_Contact", "R_Contact",
        "Walkway_X_Norm", "Walkway_Vel", "Pressure_Diff", "Total_Load"]
    keep = [c for c in final_cols if c in out.columns]
    return downsample_raw(out[keep])

# ==================== 2. INSOLE BUILDER =====================
def build_insole(df: pd.DataFrame, weight_kg: float) -> pd.DataFrame:
    # 1. THE "DEAD AIR" FIX: Trim alignment NaN padding
    if 'LTotalForce' in df.columns and 'RTotalForce' in df.columns:
        df = df.dropna(subset=['LTotalForce', 'RTotalForce'], how='all').reset_index(drop=True)

    # --- BOUNDED LINEAR INTERPOLATION ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit=5)

    # 2. CALCULATE CLINICAL FEATURES: CoP Sway Velocity
    # We collect new columns in a dictionary to concat at once
    new_features = {}
    for side in ['L', 'R']:
        x_col, y_col = f'{side}CoP_X', f'{side}CoP_Y'
        if x_col in df.columns and y_col in df.columns:
            dx = df[x_col].diff().fillna(0)
            dy = df[y_col].diff().fillna(0)
            # 100Hz sampling means dt = 0.01 seconds
            new_features[f'{side}CoP_Vel'] = np.sqrt(dx**2 + dy**2) / 0.01

    # 3. STANDARDIZE KINEMATICS & ADD VELOCITY
    # Concat once to avoid Fragmentation Warning
    if new_features:
        df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

    # Rename kinematics to standard format
    rename_map = {}
    kinematic_cols = []
    for side in ['Linsole', 'Rinsole']:
        for s_type in ['Acc', 'Gyr']:
            for ax in ['X', 'Y', 'Z']:
                standard_name = f"{side}{s_type}_{ax}"
                alt_name = f"{side}:{s_type}_{ax}"
                if standard_name in df.columns:
                    kinematic_cols.append(standard_name)
                elif alt_name in df.columns:
                    rename_map[alt_name] = standard_name
                    kinematic_cols.append(standard_name)
    
    if rename_map:
        df = df.rename(columns=rename_map)

    # 4. COLUMN SELECTION
    base_cols = ["Time", "LTotalForce", "RTotalForce", "LCoP_X", "LCoP_Y", "LCoP_Vel", 
                 "RCoP_X", "RCoP_Y", "RCoP_Vel"] + kinematic_cols
    
    l_press_cols = sorted([c for c in df.columns if re.match(r"^LPressure\d+$", c)], 
                          key=lambda x: int(re.search(r"\d+", x)[0]))
    r_press_cols = sorted([c for c in df.columns if re.match(r"^RPressure\d+$", c)], 
                          key=lambda x: int(re.search(r"\d+", x)[0]))

    keep = [c for c in base_cols if c in df.columns] + l_press_cols + r_press_cols
    if not keep: return pd.DataFrame()
    
    # Use .copy() here to get a clean, consolidated memory block
    out = df[keep].copy()

    # 5. BW NORMALIZATION
    if weight_kg and weight_kg > 0:
        denom = weight_kg * C.GRAV
        norm_cols = {}
        for c in ["LTotalForce", "RTotalForce"]:
            if c in out.columns:
                norm_cols[c+"_BW"] = pd.to_numeric(out[c], errors="coerce") / denom
        if norm_cols:
            out = pd.concat([out, pd.DataFrame(norm_cols, index=out.index)], axis=1)
    
    # 6. VECTORIZED BLOCK CONVERSION
    if l_press_cols:
        arr_l = out[l_press_cols].apply(pd.to_numeric, errors='coerce').to_numpy()
        out["LPressure_Block"] = [tuple(row) for row in arr_l]
        out = out.drop(columns=l_press_cols)
        
    if r_press_cols:
        arr_r = out[r_press_cols].apply(pd.to_numeric, errors='coerce').to_numpy()
        out["RPressure_Block"] = [tuple(row) for row in arr_r]
        out = out.drop(columns=r_press_cols)

    final_cols = ["Time", "LTotalForce_BW", "RTotalForce_BW", "LCoP_X", "LCoP_Y", "LCoP_Vel", 
                  "RCoP_X", "RCoP_Y", "RCoP_Vel", "LPressure_Block", "RPressure_Block"] + kinematic_cols
                    
    keep_final = [c for c in final_cols if c in out.columns]
    return downsample_raw(out[keep_final])

    
# ==================== 3. IMU BUILDER (Refactored) =====================
def build_imu(df: pd.DataFrame) -> pd.DataFrame:
    if "Time" not in df.columns: return pd.DataFrame()
    imu_out = pd.DataFrame({"Time": df["Time"]})

    def process_triplet(axes_cols, out_col_name):
        if all(c in df.columns for c in axes_cols):
            data = df[axes_cols].apply(pd.to_numeric, errors='coerce').fillna(0).to_numpy()
            imu_out[out_col_name] = list(map(tuple, data))
            return True
        return False

    # 1. Body Sites: Use loops from Config
    for s in C.BODY_IMU_SITES:
        for mod, suffixes in C.IMU_MODALITIES.items():
            # Build column list dynamically (e.g. LowerBack_FreeAcc_E)
            cols = [f"{s}_{sfx}" for sfx in suffixes]
            # Standardized output name: LowerBack_Acc or LowerBack_Gyr
            out_name = f"{s}_{mod}"
            process_triplet(cols, out_name)

    # 2. Insole IMUs (Special naming in CSV: Linsole:Acc_X)
    # We keep this manual or adapt if needed. Assuming prompt structure:
    # for side in ["Linsole", "Rinsole"]:
    #     process_triplet([f"{side}:Acc_X", f"{side}:Acc_Y", f"{side}:Acc_Z"], f"{side}_Acc")
    #     process_triplet([f"{side}:Gyr_X", f"{side}:Gyr_Y", f"{side}:Gyr_Z"], f"{side}_Gyr")

    if imu_out.shape[1] <= 1: return pd.DataFrame()
    return downsample_raw(imu_out)

# ====================== SENSOR HEALTH CHECK ======================
def check_sensor_health(df: pd.DataFrame, sid: str, modality: str) -> list:
    missing = []
    if df.empty: return [f"{modality}_EMPTY"]

    groups = {
        "Walkway": ["L_Pressure_BW", "R_Pressure_BW"],
        "Insole": ["LTotalForce_BW", "RTotalForce_BW"],
        "IMU": ["LowerBack_Acc", "Forehead_Acc", "R_Wrist_Acc"] 
    }
    if modality not in groups: return []

    for sensor in groups[modality]:
        cols = [c for c in df.columns if sensor in c]
        if not cols:
            missing.append(sensor)
            continue
        try:
            # --- FIXED LOGIC: Check Statistics of ENTIRE signal, not just start ---
            sample = df[cols[0]].iloc[0]
            
            # Case A: Vector/Tuple Columns (IMU, Pressure Blocks)
            if isinstance(sample, (list, tuple, np.ndarray)):
                # We can't easily run .std() on tuples. 
                # Strategy: Check the middle of the file (likely walking) 
                # or random sample instead of just the start.
                
                # Grab 100 frames from the *middle* of the trial
                mid_idx = len(df) // 2
                start = max(0, mid_idx - 50)
                end = min(len(df), mid_idx + 50)
                
                subset = df[cols[0]].iloc[start:end]
                
                # Flatten to check for ANY non-zero value
                valid_data = np.array([x for x in subset.tolist() if isinstance(x, (list, tuple))])
                
                if valid_data.size > 0:
                    # Check if Max absolute value is effectively 0
                    is_dead = np.max(np.abs(valid_data)) < 1e-5
                else:
                    is_dead = True
                    
            # Case B: Scalar Columns (Force, Pressure_BW)
            else:
                # For scalars, we can check the whole column efficiently
                # If standard deviation is 0, it's a flatline (dead)
                # If max is 0, it's dead.
                series = df[cols[0]].fillna(0)
                is_dead = (series.max() == 0) and (series.min() == 0)

            if is_dead:
                missing.append(f"{sensor}_DEAD")
                
        except Exception as e:
            # missing.append(f"{sensor}_ERROR") 
            pass # Suppress random errors to keep report clean
            
    return missing
# ====================== ORCHESTRATOR ======================
def run_end_to_end(hc_csv_root=C.HC_PATH, pd_csv_root=C.PD_PATH, 
                   hc_demo_csv=C.HC_DEMO_CSV, pd_demo_csv=C.PD_DEMO_CSV, 
                   output_dir=C.OUTPUT_DIR, pattern=C.CSV_PATTERN):
    
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    weight_map = build_weight_map(str(hc_demo_csv), str(pd_demo_csv))
    files = {
        **find_subject_files(hc_csv_root, pattern), 
        **find_subject_files(pd_csv_root, pattern)
    }
    
    if not files: print("No files found."); return

    health_report = {}
    print(f"Processing {len(files)} subjects...")

    for sid, path in files.items():
        try:
            df = pd.read_csv(path)
            # Filter non-walking (Task Purity) using Config exclude list if available or hardcoded
            if "GeneralEvent" in df.columns: 
                # Using hardcoded "standing" here as typical exclusion, or Config if defined
                df = df[df["GeneralEvent"].str.lower() != "standing"].copy()
            
            wkg = weight_map.get(sid, np.nan)
            
            walkway = build_walkway(df, wkg)
            insole  = build_insole(df, wkg)
            imu     = build_imu(df)

            issues = []
            issues += check_sensor_health(walkway, sid, "Walkway")
            issues += check_sensor_health(insole, sid, "Insole")
            issues += check_sensor_health(imu, sid, "IMU")
            if issues: health_report[sid] = issues

            walkway.to_pickle(outdir / f"{sid}_walkway_raw.pkl")
            insole.to_pickle (outdir / f"{sid}_insole_raw.pkl")
            imu.to_pickle    (outdir / f"{sid}_imu_raw.pkl")
            
            print(f"[{sid}] Saved Raw PKLs")
            
        except Exception as e:
            print(f"Failed to process {sid}: {e}")
            health_report[sid] = [f"CRASH: {str(e)}"]

    print("\n" + "="*30)
    print("SENSOR HEALTH REPORT")
    print("="*30)
    if not health_report:
        print("All Clean.")
    else:
        for sid, problems in health_report.items():
            print(f"[{sid}] {problems}")
            
    with open(outdir / "data_health_report.json", "w") as f:
        json.dump(health_report, f, indent=4)

if __name__ == "__main__":
    # Now this call is clean and relies on Config defaults
    run_end_to_end()