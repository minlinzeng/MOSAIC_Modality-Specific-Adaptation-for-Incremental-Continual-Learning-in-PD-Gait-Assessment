import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. Reproducibility
# =====================================================================
def set_deterministic_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def class_weight_tensor(counts: list, device: torch.device) -> torch.Tensor:
    w = 1.0 / (torch.tensor(counts, dtype=torch.float32, device=device) + 1e-8)
    w = w / w.sum() * len(counts)
    return w

# =====================================================================
# 2. MSBN routing and gradient lock
# =====================================================================
def set_active_task_and_freeze_fbg(model, task_id):
    """
    MSBN routing and gradient lock for FBG ResBlock1D bn1_list/bn2_list.
    """
    for m in model.modules():
        # ResBlocks with MSBN lists
        if hasattr(m, 'bn1_list') and hasattr(m, 'bn2_list'):
            for list_name in ['bn1_list', 'bn2_list']:
                bn_list = getattr(m, list_name)
                for i, bn in enumerate(bn_list):
                    if i == task_id:
                        # Current task BN: trainable + track stats
                        for param in bn.parameters():
                            param.requires_grad = True
                        bn.train() 
                    else:
                        # Historical BN: frozen stats
                        for param in bn.parameters():
                            param.requires_grad = False
                        bn.eval()



# =====================================================================
# 3. Training helpers
# =====================================================================
class EarlyStopping:
    def __init__(self, patience=15, mode='max', min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, current_score, model):
        score = current_score if self.mode == 'max' else -current_score
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            self.counter = 0
        return self.early_stop

class CurriculumScheduler:
    """
    Polynomial curriculum scheduler (WearGait-aligned, p_degree).
    """
    def __init__(self, alpha_max=0.5, kd_lambda_base=1.0, kd_lambda_min=0.1, p_degree=5.0, total_epochs=50):
        self.alpha_max = alpha_max
        self.kd_lambda_base = kd_lambda_base
        self.kd_lambda_min = kd_lambda_min
        self.p_degree = p_degree
        self.total_epochs = total_epochs

    def get_weights(self, current_epoch):
        t = current_epoch / self.total_epochs
        # Repulsive weight ramps up
        alpha = self.alpha_max * (t ** self.p_degree)
        # KD weight decays from base to min
        kd = self.kd_lambda_min + (self.kd_lambda_base - self.kd_lambda_min) * (1.0 - (t ** self.p_degree))
        return alpha, kd

# =====================================================================
# 4. CL loss components
# =====================================================================
def compute_kd_loss(logits_student, logits_teacher, tau=2.0):
    probs_teacher = F.softmax(logits_teacher / tau, dim=1)
    log_probs_student = F.log_softmax(logits_student / tau, dim=1)
    kd_loss = F.kl_div(log_probs_student, probs_teacher, reduction='batchmean') * (tau ** 2)
    return kd_loss

def compute_repulsive_loss(feat_student, feat_teacher, margin=0.3):
    """Margin repulsive topology loss"""
    cos_sim = F.cosine_similarity(feat_student, feat_teacher, dim=1)
    # ReLU: no push below margin
    raw_repulsion = F.relu(cos_sim - margin).mean()
    return raw_repulsion

# =====================================================================
# 5. Functional EWC
# =====================================================================
def compute_fisher_information(model, dataloader, device, active_mod, task_idx):
    model.eval()
    
    # Track grad-enabled params only
    names, params = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad])
    fisher_dict = {n: torch.zeros_like(p, device=device) for n, p in zip(names, params)}
    
    n_samples = 0
    print(f"      [EWC] Computing EXACT Per-Sample Empirical Fisher Matrix for {active_mod.upper()}...")
    
    for batch in dataloader:
        lin = batch['linear'].to(device) if active_mod=='linear' else None
        ang = batch['angular'].to(device) if active_mod=='angular' else None
        grf = batch['grf'].to(device) if active_mod=='grf' else None
        
        # One forward for batch predictions
        model.zero_grad(set_to_none=True)
        logits, _ = model(x_lin=lin, x_ang=ang, x_grf=grf, current_task=task_idx)
        
        log_probs = F.log_softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)  # empirical Fisher anchor
        
        current_batch_size = logits.size(0)
        
        # Per-sample gradient for Fisher
        for j in range(current_batch_size):
            model.zero_grad(set_to_none=True)
            log_prob_pred = log_probs[j, preds[j]]
            
            # autograd.grad per sample (avoid mean trap)
            grads = torch.autograd.grad(
                log_prob_pred, 
                params, 
                retain_graph=True if j < current_batch_size - 1 else False,
                allow_unused=True
            )
            
            for n, g in zip(names, grads):
                if g is not None:
                    fisher_dict[n] += g.detach() ** 2
                    
        n_samples += current_batch_size
        
    # Normalize Fisher by sample count
    for n in fisher_dict:
        fisher_dict[n] /= max(1, n_samples)
        
    return {name: f.detach().cpu().clone() for name, f in fisher_dict.items()}
    
def compute_ewc_loss(model, ewc_memories, lambda_ewc=1000.0):
    loss_ewc = 0.0
    device = next(model.parameters()).device
    
    for task_id, memory in ewc_memories.items():
        fisher = memory['fisher']
        opt_params = memory['opt_params']
        for name, param in model.named_parameters():
            if name in fisher:
                f_matrix = fisher[name].to(device)
                p_anchor = opt_params[name].to(device)
                loss_ewc += (f_matrix * (param - p_anchor) ** 2).sum()
                
    return lambda_ewc * loss_ewc