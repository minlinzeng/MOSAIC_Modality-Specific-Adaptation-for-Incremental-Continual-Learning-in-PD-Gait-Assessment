import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from itertools import cycle

import model.weargait.ewc.utility as U
from model.weargait.ewc.config import Config
from model.weargait.ewc.data_loader import prepare_split, make_sync_loaders
from model.weargait.ewc.encoder import WearGaitUniversal
from model.weargait.ewc.EWC import ElasticWeightConsolidation

# ----------------- Helper Class -----------------

class JointLoader:
    """
    Interleaves batches from multiple DataLoaders (Round-Robin).
    Strategy: 'Max-Length Cycling'. 
    - Determining Epoch Length: Based on the LARGEST dataset.
    - Imbalance Handling: Smaller datasets are cycled (repeated) until the largest finishes.
    """
    def __init__(self, loaders_dict):
        self.loaders = loaders_dict
        # 1. Don't limit sampling. Use the maximum length.
        self.max_len = max(len(l) for l in loaders_dict.values())
        self.keys = list(loaders_dict.keys())

    def __iter__(self):
        # Create iterators. Wrap smaller ones in cycle() so they never run out.
        # Note: We only cycle if len < max_len to define the behavior explicitly.
        # But simply cycling all of them and breaking at max_len is cleaner.
        iters = {}
        for k, v in self.loaders.items():
            if len(v) == self.max_len:
                iters[k] = iter(v) # The driver (largest)
            else:
                iters[k] = cycle(v) # The passengers (smaller, repeated)

        # Loop for the duration of the largest dataset
        for _ in range(self.max_len):
            # Yield one batch from EACH modality per step
            for mod in self.keys:
                yield next(iters[mod]), mod
                
    def __len__(self):
        # Total steps = (batches in largest loader) * (number of loaders)
        return self.max_len * len(self.loaders)


def train_joint_task(model, ewc, tr_loaders, val_loaders, device, epochs, num_classes, patience):
    """
    Trains on mixed batches with Class Weighting and Plateau Decay.
    """
    # 1. Loss Reweighting (Handle Class Imbalance Globally)
    # Aggregating counts across ALL training loaders
    total_counts = [0] * num_classes
    for loader in tr_loaders.values():
        # Assuming loader.dataset has a 'labels' attribute (from SingleModalityDataset)
        # We need to access the underlying dataset safely
        labels = loader.dataset.labels # SingleModalityDataset -> Subset/TensorDataset -> labels logic
        # If accessing raw labels is tricky due to subsets, we can approximate or require it passed in.
        # Fallback: Utility function usually handles this if standard structure
        pass 
    
    # Simpler calculation if passing raw counts isn't easy:
    # Just rely on U.class_weight_tensor if we can access all labels.
    # Let's gather all labels into one list for robust calculation.
    all_labels = []
    for loader in tr_loaders.values():
        all_labels.extend(loader.dataset.labels) # SingleModalityDataset exposes .labels property
        
    counts = [all_labels.count(i) for i in range(num_classes)]
    if len(all_labels) > 0:
        class_weights = U.class_weight_tensor(counts, device)
        print(f">> [Joint] Global Class Weights: {class_weights.cpu().numpy()}")
        ewc.criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        ewc.criterion = nn.CrossEntropyLoss()

    # 2. Fair Scheduler Comparison (ReduceLROnPlateau)
    # Using 'max' mode because we monitor F1 Score
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        ewc.optimizer, mode='max', factor=0.5, patience=15
    )
    early_stopper = U.EarlyStopping(patience=patience, mode='max')
    
    total_batches = sum(len(l) for l in tr_loaders.values()) # Just for logging info
    print(f">> [Joint] Training on {list(tr_loaders.keys())} | Epoch Len: {len(tr_loaders)*max(len(l) for l in tr_loaders.values())}")

    for ep in range(1, epochs+1):
        model.train()
        epoch_loss = 0.0
        
        # Mix batches dynamically (Cycling smaller sets)
        joint_iterator = JointLoader(tr_loaders)
        
        for step, ((x, y), mod) in enumerate(joint_iterator, 1):
            x, y = x.to(device), y.to(device)
            
            model.set_active_modality(mod)
            
            # Forward + Backward
            loss, _, _ = ewc.forward_backward_update(x, y)
            epoch_loss += loss
            
            # Note: ReduceLROnPlateau steps at EPOCH end, not batch end.

        # Validation
        avg_loss = epoch_loss / len(joint_iterator)
        scores = []
        breakdown = []
        
        for mod, loader in val_loaders.items():
            model.set_active_modality(mod)
            s = U.evaluate_classification(model, loader, device, metric='f1_macro')
            scores.append(s)
            breakdown.append(f"{mod}: {s:.1f}")
        
        avg_val_f1 = sum(scores) / len(scores)
        
        if ep % 5 == 0 or ep == 1:
            score_str = " | ".join(breakdown)
            current_lr = ewc.optimizer.param_groups[0]['lr']
            print(f"   Ep {ep} | Loss {avg_loss:.4f} | Avg F1: {avg_val_f1:.2f} ({score_str}) | LR: {current_lr:.6f}")

        # 3. Scheduler Step (Fair Comparison)
        scheduler.step(avg_val_f1)

        # Early Stopping
        if early_stopper(avg_val_f1, model):
            print(f"   🛑 Early Stopping at Ep {ep} (Best Avg F1: {early_stopper.best_score:.2f})")
            model.load_state_dict(early_stopper.best_model_state)
            break

    if early_stopper.best_model_state:
        model.load_state_dict(early_stopper.best_model_state)
        
    return early_stopper.best_score


def run_joint_experiment(args, data_cache, subj2label, folds):
    """
    Dedicated runner for Joint Training.
    """
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    all_mods = ["walkway", "insole", "imu"]
    log_dir = Config.CHECKPOINT_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    
    fold_scores = []

    for fi in range(len(folds)):
        print(f"\n========== Fold {fi+1}/{len(folds)} (JOINT) ==========")
        
        model = WearGaitUniversal(num_classes=args.num_classes).to(device)
        
        # Use conservative weight decay (1e-4) or matching Specialist Insole (0.05)
        # Since we mix tasks, 1e-4 is usually safer, but 0.05 is critical for Insole overfitting.
        # Strategy: Use 1e-4 globally. The Insole Encoder architecture (bottlenecks) should handle overfitting now.
        ewc = ElasticWeightConsolidation(model, nn.CrossEntropyLoss(), lr=args.lr, weight=0.0, weight_decay=1e-4)
        
        train_subs, test_subs = folds[fi]
        prep = prepare_split(train_subs, test_subs, data_cache=data_cache,
                             win=args.win_len, hop=args.hop_len, modalities=tuple(all_mods))
        tr_sync, te_sync = make_sync_loaders(prep, subj2label, batch_size=args.batch_size, num_workers=args.num_workers)

        tr_loaders = {}
        val_loaders = {}
        
        for m_idx, m in enumerate(all_mods):
            tr_loaders[m] = DataLoader(U.SingleModalityDataset(tr_sync.dataset, mod_index=m_idx),
                                       batch_size=args.batch_size, shuffle=True, num_workers=0)
            val_loaders[m] = DataLoader(U.SingleModalityDataset(te_sync.dataset, mod_index=m_idx),
                                        batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Pass num_classes for weighting calculation
        best_avg_f1 = train_joint_task(
            model, ewc, tr_loaders, val_loaders, device, 
            args.epochs, args.num_classes, args.patience
        )
        fold_scores.append(best_avg_f1)
        
        path = str(log_dir / f"ckpt_fold{fi}_JOINT.pt")
        U.save_checkpoint(model, path)
        print(f"   💾 Checkpoint saved: {path}")

    avg_score = sum(fold_scores) / len(fold_scores)
    print(f"\n✅ JOINT EXPERIMENT COMPLETE | Avg F1: {avg_score:.2f}")
    return avg_score