# JBHI26 — Multimodal Continual Learning for Gait & Movement Analysis

Official PyTorch implementation for continual learning experiments on three multimodal benchmarks:

| Dataset | Task | Modalities (arrival order) | Main script |
|---------|------|---------------------------|-------------|
| **WearGait** | HC vs. PD (binary) | `walkway` → `imu` → `insole` | `model/weargait/ewc/weargait_train2.py` |
| **FoG** | H&Y stage (3-class) | `skeleton` → `gyr` → `acc` | `model/fog/fog_train.py` |
| **FBG** | PD classification | `linear` → `angular` → `grf` | `model/fbg/fbg_cl_train.py` |

The proposed method combines **EWC**, **feature-level KD (Chimera)**, **curriculum repulsive loss**, and **modality-specific batch normalization (DBN)**. Baselines include **LwI** (optimal-transport weight fusion), **Harmony**, **DRMN**, and **MedCoSS**.

> **Note:** Update hard-coded paths under `model/*/config.py` and runner scripts before running on a new machine (see [Configuration](#configuration)).

---

## Repository structure

```
JBHI26/
├── README.md
├── requirements.txt
├── data/                          # Raw & processed data (not tracked by git)
│   ├── weargait/
│   ├── fog/
│   └── fbg/
└── model/
    ├── baselines/LwI/             # Shared optimal-transport module
    ├── weargait/ewc/              # WearGait experiments
    │   ├── weargait_train2.py     # Proposed method
    │   ├── weargait_lwi.py        # LwI baseline
    │   ├── run_ablations.py       # Ablation & baseline orchestrator
    │   ├── config.py
    │   └── ...
    ├── fog/                       # FoG experiments
    │   ├── fog_train.py
    │   ├── fog_lwi.py
    │   ├── run_abla_fog.py
    │   ├── run_fog_baselines.py
    │   └── preprocessing_fog.py
    └── fbg/                       # FBG experiments
        ├── fbg_cl_train.py
        ├── fbg_lwi.py
        ├── run_ablations_fbg.py
        └── run_fbg_baselines.py
```

Logs, checkpoints, and pretrained MedCoSS weights are written under each experiment directory at runtime and should **not** be committed to git.

---

## Requirements

- Python ≥ 3.10
- CUDA-capable GPU (recommended)
- Dependencies: see `requirements.txt`

```bash
cd /path/to/JBHI26
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Core packages: `torch`, `torchvision`, `numpy`, `pandas`, `scikit-learn`, `POT`, `cvxpy`, `matplotlib`, `openpyxl`, `c3d` (FoG preprocessing).

---

## Configuration

All paths are centralized in `model/paths.py`. By default they resolve to the original development layout (`/home/minlin/Minlin/JBHI26`), so **existing experiments are unaffected**.

To run on another machine, set the project root once:

```bash
export JBHI26_ROOT=/path/to/JBHI26
export PYTHONPATH="${JBHI26_ROOT}:${PYTHONPATH}"
```

Key path constants: `WEARGAIT_HC_RAW`, `FOG_CACHE`, `FBG_PROCESSED`, `WEARGAIT_EWC_LOG_ABLA2`, etc. (see `model/paths.py`).

All training commands below assume `PYTHONPATH` includes the repository root so that `from model....` imports resolve correctly.

---

## Data preparation

Raw datasets are **not** included in this repository due to size and licensing. Place them under `data/` as follows.

### WearGait

```
data/weargait/
├── HC_raw/                 # Healthy controls
│   ├── hc_demographic.csv
│   └── *_SelfPace_matTURN.csv
└── PD_raw/                 # Parkinson's disease
    ├── pd_demographic.csv
    └── *_SelfPace_matTURN.csv
```

Preprocessing is handled inside the training pipeline via `model/weargait/ewc/preprocessing.py` and `config.py` (30 Hz windows, 4 s / 120 frames).

### FoG

```
data/fog/
├── C3Dfiles/
├── IMU/
├── predictions/          # Pose / skeleton estimates
├── PDFEinfo.xlsx         # Labels
└── data_cache/
    └── fog_processed/    # Created by preprocessing
```

Run preprocessing once:

```bash
cd model/fog
python preprocessing_fog.py
```

### FBG

```
data/fbg/
└── Processed_FBG_Manifolds/   # Pre-built .pkl manifolds per subject
```

Use `model/fbg/preprocess_fbg.py` if you need to regenerate processed files from raw FBG recordings.

---

## Quick start

### WearGait — single run (proposed method)

```bash
cd model/weargait/ewc
export PYTHONPATH=/path/to/JBHI26:$PYTHONPATH

python -u weargait_train2.py \
  --mode cl \
  --order walkway,imu,insole \
  --seed 42 \
  --device cuda:0 \
  --ewc_lambda 5000.0 \
  --kd_lambda 1.0 \
  --repulsive_alpha 0.5 \
  --repulsive_margin 0.3 \
  --p_degree 5.0
```

### WearGait — LwI baseline

```bash
python -u weargait_lwi.py \
  --order walkway,insole,imu \
  --seed 42 \
  --device cuda:0 \
  --disable_dbn \
  --step 0.3 --step_diff 0.6 --layers 2 --kd_lambda 300.0
```

### WearGait — full ablation suite

`run_ablations.py` schedules experiments by index (1–35): core ablations, margin/gamma sweeps, and external baselines (DRMN, Harmony, LwI).

```bash
cd model/weargait/ewc

# Core ablations (Naive → EWC → Standard CL → Static Repul → Ours Full)
python -u run_ablations.py -r 1 2 3 4 5 -g 0 1

# External SOTA baselines only
python -u run_ablations.py -r 33 34 35 -g 0 1

# LwI baseline only
python -u run_ablations.py -r 35 -g 0
```

| Index | Experiment |
|-------|------------|
| 1 | Naive finetuning |
| 2 | EWC only |
| 3 | Standard CL (EWC + KD) |
| 4 | Static repulsive |
| 5 | **Ours (full)** |
| 33–35 | DRMN / Harmony / LwI |

Logs: `model/weargait/ewc/log/<experiment_name>/seed_*.out`  
Metrics CSV (proposed method): `seed_*_curves.csv`

---

### FoG

```bash
cd model/fog
export PYTHONPATH=/path/to/JBHI26:$PYTHONPATH

# Preprocess (if not done)
python preprocessing_fog.py

# Proposed method — ablations
python -u run_abla_fog.py -r 1 2 3 4 5 -g 0 1

# Baselines: Harmony, DRMN, LwI
python -u run_fog_baselines.py -g 0 1
```

Default modality order: `skeleton,gyr,acc`. Task: 3-class H&Y (`--num_classes 3`).

---

### FBG

```bash
cd model/fbg
export PYTHONPATH=/path/to/JBHI26:$PYTHONPATH

# LwI baseline
python -u run_fbg_baselines.py -g 0

# Ablations (if configured in run_ablations_fbg.py)
python -u run_ablations_fbg.py -g 0 1
```

Default modality order: `linear,angular,grf`. Point `--data_root` to your `Processed_FBG_Manifolds` directory.

---

## Method overview

**Continual modality stream.** Modalities arrive sequentially; the model must learn the new modality without forgetting previous ones.

**Proposed pipeline (`weargait_train2.py` / `fog_train.py`):**

1. **Warm-up** — train the new modality encoder with cross-entropy.
2. **EWC** — constrain important weights from prior tasks.
3. **Chimera KD** — feature distillation from the previous model.
4. **Repulsive loss** — push apart modality embeddings in shared space (with optional curriculum on margin `m` and exponent `p`).
5. **DBN** — modality-specific batch norm; disable with `--disable_dbn` for fair baseline comparison.

**LwI baseline (`*_lwi.py`):** fuses weights from the previous task into the current model via **optimal transport** (`model/baselines/LwI/optimal_transport.py`), followed by BN recalibration on the new modality data.

---

## Reproducing paper results

1. Install dependencies and place datasets as described above.
2. Fix all paths in `config.py` and runner `LOG_BASE` / `SCRIPT_DIR` variables.
3. Run orchestrators with the same seeds as in the paper (default: `2, 3, 42, 43, 44` for WearGait; `42, 43, 44, 2, 3` for FoG).
4. Aggregate logs with helper scripts under `model/weargait/ewc/log/` (e.g. `extraction.py`, `sum.py`) or `model/fog/extract_metrics.py`.

For long jobs on a server:

```bash
nohup python -u run_ablations.py -r 1 2 3 4 5 -g 0 1 > log_core.out 2>&1 &
```

---

## Citation

If you use this code, please cite our paper (details to be added upon publication):

```bibtex
@article{yourname2026jbhi,
  title   = {TBD},
  author  = {TBD},
  journal = {IEEE Journal of Biomedical and Health Informatics},
  year    = {2026}
}
```

---

## License

TBD. Contact the authors for data access and commercial use.

---

## Acknowledgements

- **LwI / optimal transport:** adapted from the LwI baseline implementation under `model/baselines/LwI/`.
- **MedCoSS:** WearGait MedCoSS scripts under `model/weargait/ewc/medcoss/` (pretrained weights are generated locally, not shipped with this repo).
