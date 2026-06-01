import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================================================================
# 1. 实验刚性约束 
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
# 2. MSBN 动态路由与梯度物理锁 (Crucial for CL)
# =====================================================================
def set_active_task_and_freeze_fbg(model, task_id):
    """
    针对 FBG 新架构的 MSBN 路由与梯度隔离锁。
    精准拦截 ResBlock1D 中的 bn1_list 和 bn2_list。
    """
    for m in model.modules():
        # 扫描定位到包含 MSBN 列表的残差块
        if hasattr(m, 'bn1_list') and hasattr(m, 'bn2_list'):
            for list_name in ['bn1_list', 'bn2_list']:
                bn_list = getattr(m, list_name)
                for i, bn in enumerate(bn_list):
                    if i == task_id:
                        # 解锁当前任务 BN 并开启 running statistics 追踪
                        for param in bn.parameters():
                            param.requires_grad = True
                        bn.train() 
                    else:
                        # 锁死历史任务 BN 并冻结统计量
                        for param in bn.parameters():
                            param.requires_grad = False
                        bn.eval()



# =====================================================================
# 3. 训练辅助组件
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
    多项式课程调度器 (Polynomial Curriculum Scheduler)
    严格对齐 WearGait，支持 p_degree 非线性控制。
    """
    def __init__(self, alpha_max=0.5, kd_lambda_base=1.0, kd_lambda_min=0.1, p_degree=5.0, total_epochs=50):
        self.alpha_max = alpha_max
        self.kd_lambda_base = kd_lambda_base
        self.kd_lambda_min = kd_lambda_min
        self.p_degree = p_degree
        self.total_epochs = total_epochs

    def get_weights(self, current_epoch):
        t = current_epoch / self.total_epochs
        # 排斥力逐渐增强
        alpha = self.alpha_max * (t ** self.p_degree)
        # KD 约束力逐渐减弱 (从 base 衰减到 min)
        kd = self.kd_lambda_min + (self.kd_lambda_base - self.kd_lambda_min) * (1.0 - (t ** self.p_degree))
        return alpha, kd

# =====================================================================
# 4. 连续学习损失函数组件 (Algorithm 1 Math Core)
# =====================================================================
def compute_kd_loss(logits_student, logits_teacher, tau=2.0):
    probs_teacher = F.softmax(logits_teacher / tau, dim=1)
    log_probs_student = F.log_softmax(logits_student / tau, dim=1)
    kd_loss = F.kl_div(log_probs_student, probs_teacher, reduction='batchmean') * (tau ** 2)
    return kd_loss

def compute_repulsive_loss(feat_student, feat_teacher, margin=0.3):
    """带边界的拓扑排斥损失"""
    cos_sim = F.cosine_similarity(feat_student, feat_teacher, dim=1)
    # ReLU 强制截断，低于 margin 的特征不再推开
    raw_repulsion = F.relu(cos_sim - margin).mean()
    return raw_repulsion

# =====================================================================
# 5. 纯函数版 EWC
# =====================================================================
def compute_fisher_information(model, dataloader, device, active_mod, task_idx):
    model.eval()
    
    # 仅追踪需要求梯度的参数，节省显存
    names, params = zip(*[(n, p) for n, p in model.named_parameters() if p.requires_grad])
    fisher_dict = {n: torch.zeros_like(p, device=device) for n, p in zip(names, params)}
    
    n_samples = 0
    print(f"      [EWC] Computing EXACT Per-Sample Empirical Fisher Matrix for {active_mod.upper()}...")
    
    for batch in dataloader:
        lin = batch['linear'].to(device) if active_mod=='linear' else None
        ang = batch['angular'].to(device) if active_mod=='angular' else None
        grf = batch['grf'].to(device) if active_mod=='grf' else None
        
        # 单次前向传播，计算整个 Batch 的推断概率
        model.zero_grad(set_to_none=True)
        logits, _ = model(x_lin=lin, x_ang=ang, x_grf=grf, current_task=task_idx)
        
        log_probs = F.log_softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1) # Empirical Fisher: 采用模型的自信预测作为锚点
        
        current_batch_size = logits.size(0)
        
        # 🌟 核心引擎：精确的逐样本 (Per-sample) 梯度抽取
        for j in range(current_batch_size):
            model.zero_grad(set_to_none=True)
            log_prob_pred = log_probs[j, preds[j]]
            
            # 使用 autograd.grad 精确计算单样本产生的真实曲率，绕过均值陷阱
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
        
    # 归一化：除以总样本数，得到真正的期望值 E[g^2]
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