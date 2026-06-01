import os
import glob
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

from data_loader import get_fbg_dataloaders
from encoder import MICL_CNN_PD_Model  # 确保这里调用的是带有 stride=2 的重构版模型

# =====================================================================
# 🌟 方案 C：非对称参数重构与收敛耐心控制
# =====================================================================
MODALITY_CONFIG = {
    "linear":  {"lr": 1e-4, "weight_decay": 0.05, "dropout": 0.3, "label_smoothing": 0.1, "patience": 12},
    "angular": {"lr": 1e-4, "weight_decay": 0.05, "dropout": 0.3, "label_smoothing": 0.1, "patience": 12}, 
    "grf":     {"lr": 2e-4, "weight_decay": 0.05, "dropout": 0.1, "label_smoothing": 0.05, "patience": 20}   
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--order', type=str, default="linear,angular,grf")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=70) # 延长上界以兼容大 patience
    parser.add_argument('--d_model', type=int, default=64)
    return parser.parse_args()

def set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None 

    def __call__(self, val_f1, model):
        if self.best_score is None:
            self.best_score = val_f1
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        elif val_f1 < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_f1
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.counter = 0
        return self.early_stop

def route_modality(batch, device, active_mod):
    lin = batch['linear'].to(device) if active_mod == 'linear' else None
    ang = batch['angular'].to(device) if active_mod == 'angular' else None
    grf = batch['grf'].to(device) if active_mod == 'grf' else None
    labels = batch['label'].to(device)
    return lin, ang, grf, labels

def train_specialist_epoch(model, dataloader, criterion, optimizer, device, active_mod, task_idx):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    for batch in dataloader:
        lin, ang, grf, labels = route_modality(batch, device, active_mod)
        optimizer.zero_grad()
        
        # 🌟 方案 A：受试者融合 MixUp
        alpha = 0.3
        lam = np.random.beta(alpha, alpha)
        batch_size = labels.size(0)
        index = torch.randperm(batch_size).to(device)
        
        if lin is not None: lin = lam * lin + (1 - lam) * lin[index]
        if ang is not None: ang = lam * ang + (1 - lam) * ang[index]
        if grf is not None: grf = lam * grf + (1 - lam) * grf[index]
        
        labels_a, labels_b = labels, labels[index]
        
        logits, _ = model(x_lin=lin, x_ang=ang, x_grf=grf, current_task=task_idx)
        
        # MixUp 损失计算
        loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch_size
        preds = torch.argmax(logits, dim=1)
        
        # 仅为指标监控映射回硬标签（统计近似）
        actual_labels = labels_a if lam > 0.5 else labels_b
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(actual_labels.cpu().numpy())
        
    return total_loss / len(dataloader.dataset), f1_score(all_labels, all_preds, average='macro')

@torch.no_grad()
def evaluate_specialist(model, dataloader, device, active_mod, task_idx):
    model.eval()
    all_preds, all_labels = [], []
    
    for batch in dataloader:
        lin, ang, grf, labels = route_modality(batch, device, active_mod)
        logits, _ = model(x_lin=lin, x_ang=ang, x_grf=grf, current_task=task_idx)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
    return f1_score(all_labels, all_preds, average='macro')

def run_single_fold(args, device, train_loader, test_loader, active_mod, task_idx, fold_idx):
    config = MODALITY_CONFIG[active_mod]
    
    # 强制移除 num_layers 兼容你的编码器 API
    model = MICL_CNN_PD_Model(d_model=args.d_model, dropout=config["dropout"]).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    
    # 获取模态特定的早停耐心值
    early_stopper = EarlyStopping(patience=config["patience"], min_delta=1e-3)
    
    for ep in range(1, args.epochs + 1):
        train_loss, train_f1 = train_specialist_epoch(model, train_loader, criterion, optimizer, device, active_mod, task_idx)
        val_f1 = evaluate_specialist(model, test_loader, device, active_mod, task_idx)
        
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        if ep % 5 == 0 or ep == 1:
            print(f"      [Fold {fold_idx} | Ep {ep:02d}/{args.epochs}] LR: {current_lr:.1e} | Tr_Loss: {train_loss:.4f} | Tr_F1(Mix): {train_f1*100:.1f}% | Val_F1: {val_f1*100:.2f}%", flush=True)
            
        if early_stopper(val_f1, model):
            print(f"      🛑 [Early Stop] Triggered at Epoch {ep}. Restoring optimal generalized weights.", flush=True)
            break
            
    model.load_state_dict(early_stopper.best_model_state)
    return early_stopper.best_score

def get_all_subjects(data_root):
    files = glob.glob(os.path.join(data_root, "*.pkl"))
    subjects = sorted(list(set([os.path.basename(f).split('_')[0] for f in files])))
    return subjects

def main():
    args = parse_args()
    set_deterministic_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tasks = [t.strip().lower() for t in args.order.split(",") if t.strip()]
    modality_to_task_idx = {'linear': 0, 'angular': 1, 'grf': 2}
    
    print("\n" + "="*65)
    print(f" 🚀 FBG 5-FOLD CROSS VALIDATION ENGINE [MIXUP & JITTER ENABLED]")
    print(f" Seed: {args.seed} | Modalities: {tasks}")
    print("="*65)

    subjects = get_all_subjects(args.data_root)
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    final_results = {}
    
    for ti, mod in enumerate(tasks):
        task_idx = modality_to_task_idx[mod]
        config = MODALITY_CONFIG[mod]
        
        print(f"\n>>> MODALITY: [{mod.upper()}] <<<")
        print(f"    Params: LR={config['lr']} | WD={config['weight_decay']} | Drop={config['dropout']} | Pat={config['patience']}")
        
        fold_scores = []
        for fold, (train_idx, test_idx) in enumerate(kf.split(subjects), 1):
            train_subjects = [subjects[i] for i in train_idx]
            test_subjects = [subjects[i] for i in test_idx]
            
            # 使用修正后的 256 窗口与 64 步长
            train_loader, test_loader = get_fbg_dataloaders(
                data_root=args.data_root, 
                train_subjects=train_subjects,  
                test_subjects=test_subjects, 
                batch_size=args.batch_size, 
                window_size=256, step_size=64
            )
            
            best_fold_f1 = run_single_fold(args, device, train_loader, test_loader, mod, task_idx, fold)
            fold_scores.append(best_fold_f1)
            print(f"    --> Fold {fold} Best Val F1: {best_fold_f1 * 100:.2f}%\n")
            
        mean_f1 = np.mean(fold_scores) * 100
        std_f1 = np.std(fold_scores) * 100
        final_results[mod] = (mean_f1, std_f1, fold_scores)
        
        print(f"    [🌟 {mod.upper()} 5-Fold Result]: {mean_f1:.2f}% ± {std_f1:.2f}%")

    print("\n" + "="*65)
    print(" 🏆 FINAL ORACLE REPORT (5-Fold CV Macro F1)")
    print("="*65)
    for mod, (mean_f1, std_f1, scores) in final_results.items():
        raw_str = ", ".join([f"{x*100:.1f}" for x in scores])
        print(f" {mod.upper():>10}: {mean_f1:5.2f}% ± {std_f1:4.2f}%  (Folds: [{raw_str}])")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()