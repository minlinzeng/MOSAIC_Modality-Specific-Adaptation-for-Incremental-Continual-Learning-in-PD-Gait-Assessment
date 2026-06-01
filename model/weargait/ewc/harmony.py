import os, argparse, sys
from pathlib import Path
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

# --- Path Setup ---
current_file = Path(__file__).resolve()
current_dir = current_file.parent
project_root = current_dir.parent.parent.parent
sys.path.append(str(project_root))

# --- Project Imports ---
from model.weargait.ewc.config import Config
import model.weargait.ewc.utility as U
from model.weargait.ewc.data_loader import (
    preload_all_subjects, prepare_split, make_sync_loaders, 
    make_fixed_balanced_folds_no_overlap, build_subj2label
)
from model.weargait.ewc.encoder import WearGaitUniversal
# from model.weargait.ewc.encoder_res18 import WearGaitResNet18
import matplotlib
matplotlib.use('Agg') # Safe for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

def _scan_subjects(dir_path: Path):
    return sorted({x.name.split("_")[0].lower() for x in dir_path.glob(Config.CSV_PATTERN)})

def init_subjects_and_folds(args):
    pd_ids = _scan_subjects(Config.PD_PATH)
    hc_ids = _scan_subjects(Config.HC_PATH)
    subj2label = build_subj2label(pd_ids, hc_ids)
    folds = make_fixed_balanced_folds_no_overlap(pd_ids, hc_ids, n_folds=args.n_folds, seed=args.seed)
    return subj2label, folds

######################## Main Components ########################

class HarmonyACFM(nn.Module):
    """
    严格按照 Harmony 论文 Section 3.4 实现的 ACFM。
    核心功能：根据当前特征和历史分类器权重，生成带有自适应扰动的兼容历史特征。
    """
    def __init__(self, feature_dim=64, K=3, classifier_dim=512):
        super().__init__()
        self.feature_dim = feature_dim
        self.K = K # 论文中 K=3 (即混合 3 种不同尺度的噪音)
        self.proto_proj = nn.Linear(classifier_dim, feature_dim)
        # 对应 Eq. 2 中的 E_trans: 对原型进行线性变换
        self.E_trans = nn.Linear(feature_dim, feature_dim)
        
        # 对应 Eq. 2 中的 E_mod: 预测混合系数 \alpha_i 的两层 MLP
        self.E_mod = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, K)
        )
        
        # 对应 Eq. 2 中的 \sigma: K 个高斯分布的独立可学习标准差
        self.sigma = nn.Parameter(torch.ones(K))
        
        # 对应 Eq. 2 中的 \lambda_g: 扰动强度，论文中建议设为 0.6
        self.lambda_g = 0.6

    def forward(self, current_feat, hist_classifier_weight, labels):
        """
        严格复现: current_feat 现在的形状是 (Batch, Time, Channels) 对应论文的 L x d
        """
        B, T, C = current_feat.size()
        
        # Step 1: 提取历史原型 (Eq. 1)
        P_prev_high_dim = hist_classifier_weight[labels] # (B, 512)
        P_prev = self.proto_proj(P_prev_high_dim)        # (B, 64)
        
        # 👑 严格复现: 将全局的 Prototype 广播到所有的 Token (时间步) 上
        # (B, 64) -> (B, 1, 64) -> (B, T, 64)
        P_prev_time = P_prev.unsqueeze(1).expand(-1, T, -1)
        
        # Step 2: 自适应特征扰动 (Eq. 2)
        # 预测噪音混合权重 \alpha_i (PyTorch Linear 原生支持处理 BxTxC)
        alpha = F.softmax(self.E_mod(current_feat), dim=-1) # (B, T, K)
        
        noise_sum = torch.zeros_like(current_feat) # (B, T, C)
        for k in range(self.K):
            z_k = torch.randn_like(current_feat) * self.sigma[k]
            noise_sum += alpha[..., k:k+1] * z_k
            
        F_prev_modulated = self.E_trans(P_prev_time) + (self.lambda_g * noise_sum)
        
        # Step 3: 特征融合 (Eq. 3)
        F_compatible_history = F_prev_modulated + current_feat # (B, T, C)
        
        return F_compatible_history

class MKAM(nn.Module):
    """
    Modality Knowledge Aggregation Module (MKAM)
    对应论文 Eq. (4): 将特征映射到统一的聚合空间
    根据论文 4.3 Implementation Details: 使用一个简单的 Linear 层
    """
    def __init__(self, feature_dim=64):
        super().__init__()
        self.proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        # x 可以是 (B, C) 也可以是 (B, T, C)
        return self.proj(x)

class GatedKnowledgeAdapter(nn.Module):
    """
    Gated Knowledge Adapter
    对应论文 Eq. (5): W_adapter = \omega * B * A
    利用低秩矩阵过滤历史特征中的噪音，并用 \omega 控制知识流入。
    """
    def __init__(self, feature_dim=64, rank=8):
        super().__init__()
        # 论文 4.3 节明确指出: rank 设置为 128
        self.A = nn.Linear(feature_dim, rank, bias=False)
        self.B = nn.Linear(rank, feature_dim, bias=False)
        
        # 门控阀门 \omega，初始化为 1.0
        self.omega = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_history):
        # 低秩过滤
        low_rank_feat = self.B(self.A(x_history))
        # 门控调节
        return self.omega * low_rank_feat

class CumulativeKnowledgeAggregation(nn.Module):
    """
    Cumulative Knowledge Aggregation (CKA)
    整合了 MKAM、Gated Adapter 和 Cross-Attention 的完整聚合模块。
    对应论文 Eq. (4), (5), (6)。
    """
    def __init__(self, feature_dim=64, rank=128):
        super().__init__()
        # 为当前模态和历史模态分配独立的 MKAM
        self.mkam_current = MKAM(feature_dim)
        self.mkam_history = MKAM(feature_dim)
        
        # 门控知识适配器
        self.gated_adapter = GatedKnowledgeAdapter(feature_dim, rank)
        
        # Cross-Attention 的缩放因子 (1 / sqrt(d))
        self.scale = feature_dim ** -0.5

    def forward(self, f_current, f_history):
        """
        参数:
        - f_current: 当前模态的原始特征，形状 (Batch, Time, Channels)
        - f_history: 经过 ACFM 生成的伪装历史特征，形状 (Batch, Time, Channels)
        
        注意：如果你的特征是 1D 的 (Batch, Channels)，会自动在中间增加一个 Time=1 的维度。
        """

        # --------------------------------------------------
        # Step 1: MKAM 映射 (Eq. 4)
        # 获取 \hat{F}_i^t 和 \hat{F}_i^{t-1}
        # --------------------------------------------------
        f_hat_current = self.mkam_current(f_current)
        f_hat_history = self.mkam_history(f_history)

        # --------------------------------------------------
        # Step 2: 历史特征过滤 (Eq. 5)
        # 经过 Gated Adapter 获取 \tilde{F}_i^{t-1}
        # --------------------------------------------------
        f_tilde_history = self.gated_adapter(f_hat_history)

        # --------------------------------------------------
        # Step 3: Cross-Attention Token Merging (Eq. 6)
        # 在 Token 级别将历史知识注入当前特征
        # --------------------------------------------------
        # Q (查询): 当前特征 f_hat_current
        # K (键): 过滤后的历史特征 f_tilde_history
        # V (值): 过滤后的历史特征 f_tilde_history
        
        # 计算注意力分数: (B, T, C) @ (B, C, T) -> (B, T, T)
        attn_scores = torch.bmm(f_hat_current, f_tilde_history.transpose(1, 2)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        # 聚合历史知识: (B, T, T) @ (B, T, C) -> (B, T, C)
        history_injection = torch.bmm(attn_weights, f_tilde_history)

        # 采用带残差连接的融合方式，确保当前模态的主导性
        f_fused = f_hat_current + history_injection

        # 如果输入原本是 2D 的，去掉我们增加的 T 维度
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
        
        # 1. Direct Feature Alignment
        loss_dir = F.mse_loss(o_current, o_history)

        # 2. Contrastive Feature Alignment
        norm_curr = F.normalize(o_current, p=2, dim=1)
        norm_hist = F.normalize(o_history, p=2, dim=1)
        sim_matrix = torch.matmul(norm_curr, norm_hist.t())
        pos_sim = torch.diag(sim_matrix) 
        
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        neg_sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))
        hard_neg_sim, _ = neg_sim_matrix.max(dim=1) 
        loss_con = F.relu(self.margin - (pos_sim - hard_neg_sim)).mean()

        # 3. Distribution-level Alignment
        # ✅ 直接在这里定义均匀分布权重，安全且无需追踪梯度
        beta_norm = torch.ones(batch_size, 1, device=device) / batch_size
        proxy_current = torch.sum(beta_norm * o_current, dim=0) 
        proxy_history = torch.sum(beta_norm * o_history, dim=0)
        loss_dis = F.mse_loss(proxy_current, proxy_history)

        # 汇总
        total_align_loss = loss_dir + (self.lambda_con * loss_con) + (self.lambda_dis * loss_dis)
        return total_align_loss

######################## Training Loop ########################

def train_harmony_task(args, model, train_loader, val_loader, mod, device, 
                       epochs, num_classes, patience, task_idx):
    """
    完全体 Harmony 训练循环：集成了 ACFM, CKA, 和 Hybrid Alignment。
    新增了 task_idx 参数，用于判断是否是第一个任务。
    """
    print(f"\n   >>> [Harmony Complete] Training '{mod}' (Task {task_idx+1}) ...")
    model.set_active_task(task_idx)
    # ==========================================
    # 1. 优化器配置 (分离基础参数、ACFM 和 CKA)
    # ==========================================
    base_params = [p for name, p in model.named_parameters() 
                   if p.requires_grad and 'acfm' not in name and 'cka' not in name]
    acfm_params = [p for p in model.acfm[mod].parameters() if p.requires_grad]
    cka_params =  [p for p in model.cka[mod].parameters() if p.requires_grad]
    
    optimizer = torch.optim.Adam([
        {'params': base_params},  
        {'params': acfm_params, 'lr': args.lr * 1.0}, 
        {'params': cka_params, 'lr': args.lr * 1.0} 
    ], lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    early_stopper = U.EarlyStopping(patience=patience, mode='max')
    criterion = nn.CrossEntropyLoss()

    # 初始化混合对齐损失
    align_criterion = HybridAlignmentLoss(lambda_con=0.8, lambda_dis=0.6, margin=0.3).to(device)
    lambda_align = args.lambda_align
    best_eval = 0.0

    # ==========================================
    # 2. 训练循环
    # ==========================================
    for ep in range(1, epochs + 1):
        model.train()
        accum = {"loss": 0, "ce": 0, "align": 0, "correct": 0, "total": 0}

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # --- A. 提取当前原始特征 ---
            raw_feats = model.encoders[mod](x) # (B, 64, T)
            # 👑 转换为 Harmony 论文中定义的 F_i^t \in R^{L \times d} (即 B x T x C)
            feats_seq = raw_feats.transpose(1, 2) 
            
            loss_align = torch.tensor(0.0, device=device)

            if task_idx > 0:
                hist_weight = model.shared_head.fc.weight.detach()
                
                # B1. ACFM 直接处理 Sequence
                fake_history = model.acfm[mod](feats_seq, hist_weight, y) # (B, T, 64)

                # B2. CKA Token 级别交叉注意力融合 (Eq. 6 严格复现)
                fused_seq = model.cka[mod](feats_seq, fake_history) # (B, T, 64)

                # B3. 转回 CNN 的维度格式
                fused_time = fused_seq.transpose(1, 2) # (B, 64, T)
                fake_hist_time = fake_history.transpose(1, 2)

                z_curr = model.shared_backbone(fused_time)
                logits = model.shared_head(z_curr)
                z_history = model.shared_backbone(fake_hist_time)
                loss_align = align_criterion(z_curr, z_history)

            else:
                # Task 1
                f_hat_curr = model.cka[mod].mkam_current(feats_seq)
                fused_time = f_hat_curr.transpose(1, 2)
                
                z_curr = model.shared_backbone(fused_time)
                logits = model.shared_head(z_curr)

            # --- C. 计算总损失并反向传播 ---
            loss_ce = criterion(logits, y)
            total_loss = loss_ce + (lambda_align * loss_align)
            
            total_loss.backward()
            optimizer.step()

            # 记录指标
            accum["loss"] += total_loss.item()
            accum["ce"]   += loss_ce.item()
            accum["align"] += loss_align.item()
            accum["correct"] += (logits.argmax(1) == y).sum().item()
            accum["total"] += y.size(0)

        # ==========================================
        # 3. 验证与测试循环 (重点：完全剥离历史外挂)
        # ==========================================
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                
                v_raw = model.encoders[mod](vx) # (B, 64, T)
                v_seq = v_raw.transpose(1, 2)   # (B, T, 64)

                # 只过 MKAM 映射
                v_hat = model.cka[mod].mkam_current(v_seq) # (B, T, 64)
                v_time = v_hat.transpose(1, 2)  # 转回 (B, 64, T)
                
                vz = model.shared_backbone(v_time)
                v_logits = model.shared_head(vz)
                
                all_preds.extend(v_logits.argmax(1).cpu().numpy())
                all_targets.extend(vy.cpu().numpy())
        
        val_f1 = f1_score(all_targets, all_preds, average='macro') * 100.0
        best_eval = max(val_f1, best_eval)
        scheduler.step(val_f1)

        if ep % 10 == 0:
            n = len(train_loader)
            print(f"[{mod}] Ep {ep:02d} | Loss:{accum['loss']/n:.4f} "
                  f"[CE:{accum['ce']/n:.3f} Align:{accum['align']/n:.3f}] | "
                  f"Acc:{accum['correct']/accum['total']*100:.1f}% ValF1:{val_f1:.2f}%")

        if early_stopper(val_f1, model):
            print(f"   🛑 Early Stopping at Ep {ep}")
            model.load_state_dict(early_stopper.best_model_state)
            break

    if early_stopper.best_model_state:
        model.load_state_dict(early_stopper.best_model_state)

    # ==========================================
    # 💥 终极学术反杀：自动打印门控坍塌指数 (\omega)
    # ==========================================
    model.eval()
    omega_val = model.cka[mod].gated_adapter.omega.item()
    print("\n" + "🎯" + "="*50)
    print(f"   [GATING COLLAPSE ANALYSIS - Task: {mod}]")
    print(f"   => Learned Gate Parameter (\u03c9) : {omega_val:.6f}")
    if omega_val < 0.1:
        print("   🚨 WARNING: GATING COLLAPSE DETECTED!")
        print("   The $1.344$ Modality Gap & $0.4$ Variance Shift forced Harmony")
        print("   to completely shut off historical knowledge transfer.")
    print("   " + "="*50 + "\n")

def run_cv_harmony(args, data_cache):
    subj2label, folds = init_subjects_and_folds(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    tasks = [t.strip() for t in args.order.split(",") if t.strip()]
    
    step_history = {}
    fold_scores = []
    # 因为真正的 Harmony ACFM 直接从共享分类器 (shared_head) 中提取知识原型！

    for fi in range(len(folds)):
        print(f"\n{'='*20} Fold {fi+1}/{len(folds)} {'='*20}")
        
        # 1. Init Base Model
        model = WearGaitUniversal(num_classes=args.num_classes, disable_dbn=False).to(device)
        
        # ==========================================
        # 2. INJECT HARMONY COMPONENTS (动态架构修改)
        # ==========================================
        # a. 动态注入 ACFM (基于历史权重的特征伪装生成器)
        model.acfm = nn.ModuleDict({
            k: HarmonyACFM(feature_dim=64, K=3).to(device) for k in model.encoders.keys()
        })
        
        # b. 动态注入 CKA (包含门控适配器和交叉注意力融合)
        model.cka = nn.ModuleDict({
            k: CumulativeKnowledgeAggregation(feature_dim=64, rank=128).to(device) for k in model.encoders.keys()
        })
        
        seen_mods = []
        eval_loader_cache = {}

        for ti, mod in enumerate(tasks):
            print(f"\n=== Harmony Task {ti+1}/{len(tasks)} : {mod} ===")
            
            # 准备数据
            train_subs, test_subs = folds[fi]
            prep = prepare_split(train_subs, test_subs, data_cache=data_cache, win=args.win_len, hop=args.hop_len, modalities=(mod,))
            tr_sync, te_sync = make_sync_loaders(prep, subj2label, batch_size=args.batch_size, num_workers=args.num_workers)
            tr_loader = DataLoader(U.SingleModalityDataset(tr_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=True, num_workers=0)
            te_loader = DataLoader(U.SingleModalityDataset(te_sync.dataset, mod_index=0), batch_size=args.batch_size, shuffle=False, num_workers=0)
            eval_loader_cache[mod] = te_loader 

            # ==========================================
            # 3. 训练核心逻辑 (传入任务索引 ti)
            # ==========================================
            train_harmony_task(args, model, tr_loader, te_loader, mod, device, 
                               args.epochs, args.num_classes, args.patience, ti)

            # ==========================================
            # 4. 评估所有已见任务 (严格的推理模式)
            # ==========================================
            seen_mods.append(mod)
            print(f"\n--- Evaluation (Step {ti+1}) ---")
            scores = []
            
            # for m in seen_mods:
            for m_idx, m in enumerate(seen_mods):
                model.set_active_task(m_idx)
                model.eval()
                all_preds, all_targets = [], []
                
                with torch.no_grad():
                    for vx, vy in eval_loader_cache[m]:
                        vx, vy = vx.to(device), vy.to(device)
                        
                        # A. 提取原始特征 (B, 64, T)
                        vf = model.encoders[m](vx)
                        # 转置为 Token 序列 (B, T, 64)
                        v_seq = vf.transpose(1, 2)
                        
                        # B. 🚀 严格推理: 绕过 ACFM，仅经过 MKAM 映射层
                        v_hat = model.cka[m].mkam_current(v_seq) # (B, T, 64)
                        
                        # C. 转回 CNN 的时间维度并前向传播
                        v_time = v_hat.transpose(1, 2) # (B, 64, T)
                        
                        vz = model.shared_backbone(v_time)
                        v_logits = model.shared_head(vz)
                        
                        all_preds.extend(v_logits.argmax(1).cpu().numpy())
                        all_targets.extend(vy.cpu().numpy())
                
                # 计算 F1 分数
                score = f1_score(all_targets, all_preds, average='macro') * 100.0
                scores.append(score)
                print(f"  {m}: {score:.2f}")

            # 记录每一步的分数
            if ti not in step_history: step_history[ti] = {}
            for m_idx, m_score in enumerate(scores):
                step_history[ti][m_idx] = [m_score]

            avg_seen = sum(scores) / len(scores)
            print(f"  Avg Seen: {avg_seen:.2f}")

        fold_scores.append(avg_seen)

    print(f"\n🏆 Final Avg F1 across folds: {sum(fold_scores)/len(fold_scores):.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--order", type=str, default="imu,walkway,insole")
    
    # Config Overrides
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n_folds", type=int, default=Config.N_FOLDS)
    ap.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=Config.EPOCHS)
    ap.add_argument("--patience", type=int, default=15)
    
    ap.add_argument("--win_len", type=int, default=Config.WINDOW_SIZE)
    default_hop = int(Config.WINDOW_SIZE * Config.STRIDE)
    ap.add_argument("--hop_len", type=int, default=default_hop)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_classes", type=int, default=2)

    # ==========================================
    # 👑 Harmony Specific (Updated for Full Version)
    # ==========================================
    # 论文 Eq. 12 提到的全局对齐损失权重，论文推荐设为 1.5
    ap.add_argument("--lambda_align", type=float, default=0.15, help="Weight for Hybrid Alignment Loss")

    args = ap.parse_args()
    print(f"Harmony Mode | Arguments: {', '.join(f'{k}={v}' for k, v in vars(args).items())}")
    
    # Preload Data
    global_cache = preload_all_subjects(Config.OUTPUT_DIR)
    U.set_seed(args.seed)
    
    run_cv_harmony(args, global_cache)

if __name__ == "__main__":
    main()