import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# --- 导入 FBG 资产 ---
from data_loader import get_fbg_dataloaders
from encoder import ResBlock1D, MICL_CNN_PD_Model  
from fbg_utility import set_deterministic_seed, EarlyStopping
from model.paths import FBG_PROCESSED, as_str

# =====================================================================
# 🌟 1. FBG 架构适配器 (Harmony Wrapper)
# =====================================================================
class FBGEncoderWrapper(nn.Module):
    """提取单个模态的前端和 Dropout"""
    def __init__(self, encoder, input_drop):
        super().__init__()
        self.encoder = encoder
        self.input_drop = input_drop
    def forward(self, x):
        # x: (B, T, C) -> permute -> Drop -> Conv -> (B, 64, T_down)
        return self.encoder(self.input_drop(x.permute(0, 2, 1)))

class FBGSharedBackbone(nn.Module):
    """封装残差块，屏蔽 current_task 参数 (因为 Harmony 是基于历史特征聚合，不是 MSBN)"""
    def __init__(self, res1, res2, res3):
        super().__init__()
        self.res1 = res1
        self.res2 = res2
        self.res3 = res3
    def forward(self, x):
        # Harmony 基线不使用 MSBN，所以 task_id 固定传 0
        x = self.res1(x, task_id=0)
        x = self.res2(x, task_id=0)
        x = self.res3(x, task_id=0)
        return torch.mean(x, dim=2) # Pooled output (B, 32)

class FBGSharedHead(nn.Module):
    """封装分类头和噪声"""
    def __init__(self, head, dropout, noise_std):
        super().__init__()
        # Harmony 原代码提取权重是 model.shared_head.fc.weight，所以我们包一层 fc
        self.fc = head 
        self.dropout = dropout
        self.noise_std = noise_std
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return self.fc(self.dropout(x))

class HarmonyFBGModel(nn.Module):
    """
    将 MICL_CNN_PD_Model 拆解组装成 Harmony 期待的接口格式。
    !!! 注意: 基线必须禁用 MSBN !!!
    """
    def __init__(self, d_model=64, num_tasks=3, dropout=0.3):
        super().__init__()
        # 实例化底层模型 (强制禁用 MSBN，作为纯净基线)
        self.base_model = MICL_CNN_PD_Model(d_model=d_model, num_tasks=num_tasks, dropout=dropout, disable_msbn=True)
        
        # 组装 Harmony 接口
        self.encoders = nn.ModuleDict({
            'linear': FBGEncoderWrapper(self.base_model.enc_lin, self.base_model.input_drop),
            'angular': FBGEncoderWrapper(self.base_model.enc_ang, self.base_model.input_drop),
            'grf': FBGEncoderWrapper(self.base_model.enc_grf, self.base_model.input_drop)
        })
        
        self.shared_backbone = FBGSharedBackbone(self.base_model.res1, self.base_model.res2, self.base_model.res3)
        self.shared_head = FBGSharedHead(self.base_model.head, self.base_model.dropout, self.base_model.noise_std)
        
        # 这些将在 run_cv_harmony 中动态注入
        self.acfm = None
        self.cka = None

    def set_active_task(self, task_idx):
        # 兼容 WearGait 代码中的调用，虽然在 FBG 基线中我们禁用了 MSBN，但保留接口防止报错
        pass


# =====================================================================
# 🌟 2. 严格复制的 Harmony 核心算法 (100% 保持不变)
# =====================================================================
class HarmonyACFM(nn.Module):
    def __init__(self, feature_dim=64, K=3, classifier_dim=32): # 注意: FBG 分类头输入是 32 维
        super().__init__()
        self.feature_dim = feature_dim
        self.K = K
        self.proto_proj = nn.Linear(classifier_dim, feature_dim)
        self.E_trans = nn.Linear(feature_dim, feature_dim)
        self.E_mod = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, K)
        )
        self.sigma = nn.Parameter(torch.ones(K))
        self.lambda_g = 0.6

    def forward(self, current_feat, hist_classifier_weight, labels):
        B, T, C = current_feat.size()
        P_prev_high_dim = hist_classifier_weight[labels]
        P_prev = self.proto_proj(P_prev_high_dim)
        P_prev_time = P_prev.unsqueeze(1).expand(-1, T, -1)
        
        alpha = F.softmax(self.E_mod(current_feat), dim=-1)
        noise_sum = torch.zeros_like(current_feat)
        for k in range(self.K):
            z_k = torch.randn_like(current_feat) * self.sigma[k]
            noise_sum += alpha[..., k:k+1] * z_k
            
        F_prev_modulated = self.E_trans(P_prev_time) + (self.lambda_g * noise_sum)
        F_compatible_history = F_prev_modulated + current_feat
        return F_compatible_history

class MKAM(nn.Module):
    def __init__(self, feature_dim=64):
        super().__init__()
        self.proj = nn.Linear(feature_dim, feature_dim)
    def forward(self, x):
        return self.proj(x)

class GatedKnowledgeAdapter(nn.Module):
    def __init__(self, feature_dim=64, rank=8): # 保持你原来的 rank 默认参数
        super().__init__()
        self.A = nn.Linear(feature_dim, rank, bias=False)
        self.B = nn.Linear(rank, feature_dim, bias=False)
        self.omega = nn.Parameter(torch.tensor(1.0))
    def forward(self, x_history):
        low_rank_feat = self.B(self.A(x_history))
        return self.omega * low_rank_feat

class CumulativeKnowledgeAggregation(nn.Module):
    def __init__(self, feature_dim=64, rank=128):
        super().__init__()
        self.mkam_current = MKAM(feature_dim)
        self.mkam_history = MKAM(feature_dim)
        self.gated_adapter = GatedKnowledgeAdapter(feature_dim, rank)
        self.scale = feature_dim ** -0.5

    def forward(self, f_current, f_history):
        f_hat_current = self.mkam_current(f_current)
        f_hat_history = self.mkam_history(f_history)
        f_tilde_history = self.gated_adapter(f_hat_history)
        
        attn_scores = torch.bmm(f_hat_current, f_tilde_history.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        history_injection = torch.bmm(attn_weights, f_tilde_history)
        f_fused = f_hat_current + history_injection

        if f_fused.size(1) == 1:
            f_fused = f_fused.squeeze(1)
        return f_fused

class HybridAlignmentLoss(nn.Module):
    def __init__(self, lambda_con=0.8, lambda_dis=0.6, margin=0.3):
        super().__init__()
        self.lambda_con = lambda_con
        self.lambda_dis = lambda_dis
        self.margin = margin

    def forward(self, o_current, o_history):
        batch_size = o_current.size(0)
        device = o_current.device
        loss_dir = F.mse_loss(o_current, o_history)

        norm_curr = F.normalize(o_current, p=2, dim=1)
        norm_hist = F.normalize(o_history, p=2, dim=1)
        sim_matrix = torch.matmul(norm_curr, norm_hist.t())
        pos_sim = torch.diag(sim_matrix) 
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        neg_sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        hard_neg_sim, _ = neg_sim_matrix.max(dim=1) 
        loss_con = F.relu(self.margin - (pos_sim - hard_neg_sim)).mean()

        beta_norm = torch.ones(batch_size, 1, device=device) / batch_size
        proxy_current = torch.sum(beta_norm * o_current, dim=0) 
        proxy_history = torch.sum(beta_norm * o_history, dim=0)
        loss_dis = F.mse_loss(proxy_current, proxy_history)

        total_align_loss = loss_dir + (self.lambda_con * loss_con) + (self.lambda_dis * loss_dis)
        return total_align_loss


# =====================================================================
# 🌟 3. FBG 定制的训练循环与数据路由
# =====================================================================
def train_harmony_task(args, model, train_loader, val_loader, mod, device, epochs, patience, task_idx):
    print(f"\n   >>> [Harmony Complete] Training '{mod}' (Task {task_idx+1}) ...")
    
    # 动态配置各模态的专属参数 (参考你 FBG 的 MODALITY_CONFIG)
    lr = 2e-4 if mod == 'grf' else 1e-4
    
    base_params = [p for name, p in model.named_parameters() 
                   if p.requires_grad and 'acfm' not in name and 'cka' not in name]
    acfm_params = [p for p in model.acfm[mod].parameters() if p.requires_grad]
    cka_params =  [p for p in model.cka[mod].parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam([
        {'params': base_params},  
        {'params': acfm_params, 'lr': lr * 1.0}, 
        {'params': cka_params, 'lr': lr * 1.0} 
    ], lr=lr, weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopper = EarlyStopping(patience=patience, min_delta=1e-4)
    criterion = nn.CrossEntropyLoss()
    align_criterion = HybridAlignmentLoss(lambda_con=0.8, lambda_dis=0.6, margin=0.3).to(device)

    for ep in range(1, epochs + 1):
        model.train()
        accum = {"loss": 0, "ce": 0, "align": 0, "correct": 0, "total": 0}

        # 🌟 修改点 1：解包 FBG 的字典格式
        for batch in train_loader:
            x = batch[mod].to(device)
            y = batch['label'].to(device)
            
            optimizer.zero_grad()

            raw_feats = model.encoders[mod](x) # (B, 64, T)
            feats_seq = raw_feats.transpose(1, 2) # (B, T, 64)
            
            loss_align = torch.tensor(0.0, device=device)

            if task_idx > 0:
                hist_weight = model.shared_head.fc.weight.detach()
                fake_history = model.acfm[mod](feats_seq, hist_weight, y)
                fused_seq = model.cka[mod](feats_seq, fake_history)

                fused_time = fused_seq.transpose(1, 2)
                fake_hist_time = fake_history.transpose(1, 2)

                z_curr = model.shared_backbone(fused_time)
                logits = model.shared_head(z_curr)
                z_history = model.shared_backbone(fake_hist_time)
                loss_align = align_criterion(z_curr, z_history)

            else:
                f_hat_curr = model.cka[mod].mkam_current(feats_seq)
                fused_time = f_hat_curr.transpose(1, 2)
                z_curr = model.shared_backbone(fused_time)
                logits = model.shared_head(z_curr)

            loss_ce = criterion(logits, y)
            total_loss = loss_ce + (args.lambda_align * loss_align)
            
            total_loss.backward()
            optimizer.step()

            accum["loss"] += total_loss.item()
            accum["ce"] += loss_ce.item()
            accum["align"] += loss_align.item()
            accum["correct"] += (logits.argmax(1) == y).sum().item()
            accum["total"] += y.size(0)

        # --- FBG 验证逻辑 ---
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                vx = batch[mod].to(device)
                vy = batch['label'].to(device)
                
                v_raw = model.encoders[mod](vx)
                v_seq = v_raw.transpose(1, 2)
                v_hat = model.cka[mod].mkam_current(v_seq)
                v_time = v_hat.transpose(1, 2)
                
                vz = model.shared_backbone(v_time)
                v_logits = model.shared_head(vz)
                
                all_preds.extend(v_logits.argmax(1).cpu().numpy())
                all_targets.extend(vy.cpu().numpy())
        
        val_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
        scheduler.step(val_f1)

        if ep % 5 == 0 or ep == 1:
            n = len(train_loader)
            print(f"      [Ep {ep:02d}] Tr_Loss: {accum['loss']/n:.3f} | Tr_Acc: {accum['correct']/accum['total']*100:.1f}% | Val_F1: {val_f1:.2f}%")

        if early_stopper(val_f1, model):
            print(f"      🛑 Convergence: Early Stop at Ep {ep}.")
            break

    model.load_state_dict(early_stopper.best_model_state)
    model.eval()
    omega_val = model.cka[mod].gated_adapter.omega.item()
    print(f"      [GATING COLLAPSE ANALYSIS] Gate (\u03c9): {omega_val:.6f}")


def get_all_subjects(data_root):
    import glob
    files = glob.glob(os.path.join(data_root, "*.pkl"))
    subjects = sorted(list(set([os.path.basename(f).split('_')[0] for f in files])))
    return subjects

def run_cv_harmony(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tasks = [t.strip().lower() for t in args.order.split(",")]
    num_tasks = len(tasks)
    
    subjects = get_all_subjects(args.data_root)
    kf = KFold(n_splits=5, shuffle=True, random_state=args.seed)
    
    R_matrix = np.zeros((5, num_tasks, num_tasks))
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(subjects)):
        print(f"\n{'='*60}\n 🌟 INITIATING FOLD {fold+1}/5 \n{'='*60}")
        train_subjects = [subjects[i] for i in train_idx]
        test_subjects = [subjects[i] for i in test_idx]
        
        # 🌟 使用 FBG 专属大字典 Dataloader
        train_loader, test_loader = get_fbg_dataloaders(
            args.data_root, train_subjects, test_subjects, 
            batch_size=args.batch_size, window_size=args.window_size, step_size=args.step_size
        )
        
        model = HarmonyFBGModel(d_model=64, num_tasks=num_tasks).to(device)
        
        # 动态注入 Harmony 组件
        model.acfm = nn.ModuleDict({
            k: HarmonyACFM(feature_dim=64, K=3, classifier_dim=32).to(device) for k in model.encoders.keys()
        })
        model.cka = nn.ModuleDict({
            k: CumulativeKnowledgeAggregation(feature_dim=64, rank=128).to(device) for k in model.encoders.keys()
        })
        
        seen_mods = []
        for task_idx, active_mod in enumerate(tasks):
            patience = 20 if active_mod == 'grf' else 15
            train_harmony_task(args, model, train_loader, test_loader, active_mod, device, 
                               args.epochs, patience, task_idx)

            seen_mods.append(active_mod)
            print(f"\n      --- Evaluation (Post-{active_mod.upper()}) ---")
            
            for j, eval_mod in enumerate(seen_mods):
                model.eval()
                all_preds, all_targets = [], []
                
                with torch.no_grad():
                    for batch in test_loader:
                        vx = batch[eval_mod].to(device)
                        vy = batch['label'].to(device)
                        
                        v_raw = model.encoders[eval_mod](vx)
                        v_seq = v_raw.transpose(1, 2)
                        v_hat = model.cka[eval_mod].mkam_current(v_seq)
                        v_time = v_hat.transpose(1, 2)
                        
                        vz = model.shared_backbone(v_time)
                        v_logits = model.shared_head(vz)
                        
                        all_preds.extend(v_logits.argmax(1).cpu().numpy())
                        all_targets.extend(vy.cpu().numpy())
                
                f1_score_j = f1_score(all_targets, all_preds, average='macro') * 100.0
                R_matrix[fold, task_idx, j] = f1_score_j
                print(f"      [EVAL] Testing {eval_mod.upper()}: {f1_score_j:.2f}%")

    print("\n" + "="*60 + "\n 🏆 FINAL METRIC MATRIX (R_N,N) \n" + "="*60)
    mean_R = np.mean(R_matrix, axis=0)
    std_R = np.std(R_matrix, axis=0)
    
    print("      [" + "]\t[".join([t.upper()[:3] for t in tasks]) + "]")
    for i in range(num_tasks):
        row_str = f"T{i}:  "
        for j in range(num_tasks):
            if j <= i:
                row_str += f"{mean_R[i,j]:.1f}±{std_R[i,j]:.1f}\t"
            else:
                row_str += "-----\t"
        print(row_str)
        
    bwt = np.mean([mean_R[-1, j] - mean_R[j, j] for j in range(num_tasks - 1)])
    avg_acc = np.mean(mean_R[-1, :])
    print(f"\nFinal Average F1 (A_N): {avg_acc:.2f}%")
    print(f"Backward Transfer (BWT): {bwt:.2f}%")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', type=str, default=as_str(FBG_PROCESSED))
    ap.add_argument('--order', type=str, default="linear,angular,grf")
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=50) # 对齐 FBG 的 70 轮
    
    # 🌟 强制物理时间窗对齐
    ap.add_argument('--window_size', type=int, default=256)
    ap.add_argument('--step_size', type=int, default=64)
    
    # Harmony 专属
    ap.add_argument("--lambda_align", type=float, default=0.15)
    
    args = ap.parse_args()
    set_deterministic_seed(args.seed)
    
    run_cv_harmony(args)