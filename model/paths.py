"""
Central path configuration for JBHI26.

Override the project root without changing code:
    export JBHI26_ROOT=/path/to/JBHI26

If unset, defaults to the original development machine path so
ongoing experiments keep working unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path

# Fallback preserves existing local layout when JBHI26_ROOT is not set.
_DEFAULT_ROOT = Path("/home/minlin/Minlin/JBHI26")


def project_root() -> Path:
    return Path(os.environ.get("JBHI26_ROOT", _DEFAULT_ROOT)).expanduser().resolve()


ROOT = project_root()
DATA = ROOT / "data"
MODEL = ROOT / "model"

# --- WearGait ---
WEARGAIT_DATA = DATA / "weargait"
WEARGAIT_HC_RAW = WEARGAIT_DATA / "HC_raw"
WEARGAIT_PD_RAW = WEARGAIT_DATA / "PD_raw"
WEARGAIT_EWC = MODEL / "weargait" / "ewc"
WEARGAIT_CHECKPOINTS = WEARGAIT_EWC / "checkpoints"
WEARGAIT_EWC_LOG = WEARGAIT_EWC / "log"
WEARGAIT_EWC_LOG_ABLA2 = WEARGAIT_EWC_LOG / "abla2"
WEARGAIT_MEDCOSS_LOG = WEARGAIT_EWC / "medcoss" / "log"

# --- FoG ---
FOG_DATA = DATA / "fog"
FOG_C3D = FOG_DATA / "C3Dfiles"
FOG_IMU = FOG_DATA / "IMU"
FOG_POSE = FOG_DATA / "predictions"
FOG_LABELS = FOG_DATA / "PDFEinfo.xlsx"
FOG_CACHE = FOG_DATA / "data_cache" / "fog_processed"
FOG_MODEL = MODEL / "fog"
FOG_BASELINES_LOG = FOG_MODEL / "log" / "fog_baselines"

# --- FBG ---
FBG_DATA = DATA / "fbg"
FBG_C3D = FBG_DATA / "C3Dfiles"
FBG_PROCESSED = FBG_DATA / "Processed_FBG_Manifolds"

# --- Other datasets (analysis scripts) ---
DRIVE_ACT_DATA = DATA / "drive&act"


def as_str(path: Path) -> str:
    return str(path)
