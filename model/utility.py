# file: utility.py
# Common helpers for TRIP/FOCAL training. Zero drama, just functions.

from __future__ import annotations
import os
import random
from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import torch
import torch.nn.functional as F
import copy


# ----------------------- RNG / misc -----------------------

def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set RNG seeds across libs. Optionally force deterministic CUDA behavior."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Uncomment if you need hard determinism on some CUDA ops:
        # torch.use_deterministic_algorithms(True)
        # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def flatten_skel(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """
    If skeleton is (B, T, J, C), flatten joints to (B, T, J*C).
    Leaves other shapes untouched. Returns None if x is None.
    """
    if x is None:
        return None
    if x.dim() == 4:
        b, t, j, c = x.shape
        return x.reshape(b, t, j * c)  # reshape handles non-contiguous safely
    return x


# ----------------------- folds / counts -----------------------

def generate_class_stratified_folds(reader: Any, dataset: str,
                                    exclude_subjects: Optional[List[str]] = None
                                    ) -> List[Tuple[List[str], List[str]]]:
    """
    Build subject-wise folds with class balance. Returns list of (train_subjects, eval_subjects).
    - walk: only subjects that truly have BOTH modalities end up in eval pools.
    - turn: uses reader.labels_dict; optional explicit exclusions.
    """
    exclude_subjects = set(exclude_subjects or [])

    if dataset == "walk":
        pose_pfx = {"_".join(k.split("_")[:2]) for k in reader.pose_dict.keys()}
        sens_pfx = {"_".join(k.split("_")[:2]) for k in reader.sensor_dict.keys()}
        both_modal = pose_pfx & sens_pfx
        # label per subject prefix from pose labels, filtered to both_modal
        raw = reader.pose_label_dict  # {subj_prefix: label}
        label_dict = {s: raw[s] for s in raw if s in both_modal and s not in exclude_subjects}

    elif dataset == "turn":
        # reader.labels_dict: {SUBXX: (label, ...)}
        label_dict = {
            s: reader.labels_dict[s][0]
            for s in reader.labels_dict
            if s not in exclude_subjects
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # group by class
    by_cls: Dict[int, List[str]] = {}
    for subj, y in label_dict.items():
        by_cls.setdefault(int(y), []).append(subj)

    # smallest class size decides num folds
    m = min(len(v) for v in by_cls.values())
    if m == 0:
        raise ValueError("At least one class has zero usable subjects.")

    # shuffle within class for stability (but reproducibility depends on caller setting seed)
    for k in by_cls:
        random.shuffle(by_cls[k])

    folds: List[Tuple[List[str], List[str]]] = []
    for i in range(m):
        eval_subj = [by_cls[k][i] for k in sorted(by_cls)]
        train_subj = [s for s in label_dict if s not in eval_subj]
        folds.append((train_subj, eval_subj))
    return folds


def get_branch_class_counts(loader, num_classes: int) -> Tuple[List[int], List[int]]:
    """
    Iterate a DataLoader that yields dict batches with optional 'label_skeleton' and 'label_sensor'.
    Returns two lists of length C with per-class counts for each branch.
    """
    from collections import Counter
    sk = Counter()
    se = Counter()
    for batch in loader:
        if "label_skeleton" in batch and batch["label_skeleton"] is not None:
            sk.update(torch.as_tensor(batch["label_skeleton"]).cpu().tolist())
        if "label_sensor" in batch and batch["label_sensor"] is not None:
            se.update(torch.as_tensor(batch["label_sensor"]).cpu().tolist())
    s_list = [sk[i] for i in range(num_classes)]
    e_list = [se[i] for i in range(num_classes)]
    return s_list, e_list


def class_weight_tensor(counts: List[int], device: torch.device) -> torch.Tensor:
    """
    Inverse-frequency weights normalized to sum to C. Safe for zeros via epsilon.
    """
    w = 1.0 / (torch.tensor(counts, dtype=torch.float32, device=device) + 1e-8)
    w = w / w.sum() * len(counts)
    return w


def print_class_balance(loader, num_classes, tag="EVAL", label_names=None):
    """
    Shows per-class counts and percentages for skeleton and sensor labels
    as they actually appear in the given DataLoader.
    Returns (counts_skel, counts_sensor) as two lists of length C.
    """
    from collections import Counter
    import torch

    sk = Counter(); se = Counter()
    for batch in loader:
        if "label_skeleton" in batch and batch["label_skeleton"] is not None:
            sk.update(torch.as_tensor(batch["label_skeleton"]).cpu().tolist())
        if "label_sensor" in batch and batch["label_sensor"] is not None:
            se.update(torch.as_tensor(batch["label_sensor"]).cpu().tolist())

    names = label_names or [str(i) for i in range(num_classes)]
    tot_sk, tot_se = sum(sk.values()), sum(se.values())

    print(f"\n[{tag}] class balance")
    print("class   skel_cnt  skel_%    sens_cnt  sens_%")
    for i, nm in enumerate(names):
        csk = sk[i];  cse = se[i]
        psk = (csk / tot_sk * 100.0) if tot_sk else 0.0
        pse = (cse / tot_se * 100.0) if tot_se else 0.0
        print(f"{nm:>5}   {csk:9d}  {psk:6.1f}%   {cse:9d}  {pse:6.1f}%")

    sk_list = [sk[i] for i in range(num_classes)]
    se_list = [se[i] for i in range(num_classes)]
    return sk_list, se_list


def count_params(m, trainable_only=True):
    ps = [p.numel() for p in m.parameters() if (p.requires_grad or not trainable_only)]
    return sum(ps)

# ----------------------- eval / ensemble -----------------------

def ensemble_logits(logits_a: torch.Tensor, logits_b: torch.Tensor,
                    method: str = "prob_mean") -> torch.Tensor:
    """
    Combine two classifier outputs.
    - 'prob_mean': average softmax probabilities then argmax later
    - 'logit_sum': sum raw logits (more robust under class imbalance)
    Returns combined logits (same shape as inputs).
    """
    if method == "prob_mean":
        pa = F.softmax(logits_a, dim=1)
        pb = F.softmax(logits_b, dim=1)
        return (pa + pb) * 0.5
    elif method == "logit_sum":
        return logits_a + logits_b
    else:
        raise ValueError(f"Unknown ensemble method: {method}")


def accuracy_from_logits(logits: Optional[torch.Tensor],
                         labels: Optional[torch.Tensor]) -> float:
    """Compute accuracy in percent; returns 0 if inputs are None or empty."""
    if logits is None or labels is None or labels.numel() == 0:
        return 0.0
    preds = logits.argmax(1)
    return (preds == labels).float().mean().item() * 100.0


def _safe_report(trues, preds, label_names=None, name=""):
    """Pretty print a report+confusion matrix if lists are non-empty."""
    trues = list(trues or [])
    preds = list(preds or [])
    if not trues or not preds:
        print(f"\n{name}: (no samples)")
        return
    # use label names only if counts match
    tn = (label_names if (label_names and len(set(trues)) == len(label_names)) else None)
    print(f"\n{name} Report:")
    print(classification_report(trues, preds, digits=2, zero_division=0, target_names=tn))
    print(f"{name} Confusion Matrix:")
    print(confusion_matrix(trues, preds))

def print_eval_matrix(best: dict, synced: bool, label_names=None, prefix: str = ""):
    """
    Print eval report(s) from the `best` dict collected in run_epoch.
    - synced=True  → one shared head (uses T_sk/P_sk)
    - synced=False → separate skeleton & sensor heads
    """
    if prefix:
        print(prefix)
    if synced:
        _safe_report(best.get("T_sk"), best.get("P_sk"), label_names, "Shared Head")
    else:
        _safe_report(best.get("T_sk"), best.get("P_sk"), label_names, "Skeleton Head")
        _safe_report(best.get("T_se"), best.get("P_se"), label_names, "Sensor   Head")


# ----------------------- convenience I/O -----------------------

def save_checkpoint(model: torch.nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

# [Append this to the end of utility.py]

# ----------------------- Training Helpers -----------------------

class EarlyStopping:
    """Stops training if metric doesn't improve after a given patience."""
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, current_score, model):
        if self.best_score is None:
            self.best_score = current_score
            self.save_checkpoint(model)
        else:
            if self.mode == 'max': 
                improvement = current_score - self.best_score
            else: 
                improvement = self.best_score - current_score

            if improvement > self.min_delta:
                self.best_score = current_score
                self.save_checkpoint(model)
                self.counter = 0
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        self.best_model_state = copy.deepcopy(model.state_dict())


class SingleModalityDataset(torch.utils.data.Dataset):
    """
    Generic Wrapper: extracting a single tensor from a Dict-based dataset.
    Assumes base dataset returns {'xs': [t1, t2...], 'y': label}.
    """
    def __init__(self, base_ds, mod_index=0):
        super().__init__()
        self.base = base_ds
        self.mod_index = mod_index
        # Try to expose labels if available (for class weighting)
        if hasattr(self.base, 'labels'):
            self.labels = self.base.labels
        elif hasattr(self.base, 'y'):
             self.labels = self.base.y

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        b = self.base[i]
        # Extract the specific modality tensor and label
        x = b["xs"][self.mod_index]         
        y = b["y"]
        return x, y


@torch.no_grad()
def evaluate_classification(model, loader, device, metric='f1_macro'):
    """
    Standard eval loop for classification. 
    Returns the score (e.g. Macro F1 * 100).
    """
    from sklearn.metrics import f1_score, accuracy_score
    
    model.eval()
    all_preds, all_targets = [], []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y.cpu().numpy())
    
    if metric == 'f1_macro':
        return f1_score(all_targets, all_preds, average='macro') * 100.0
    elif metric == 'accuracy':
        return accuracy_score(all_targets, all_preds) * 100.0
    else:
        return 0.0