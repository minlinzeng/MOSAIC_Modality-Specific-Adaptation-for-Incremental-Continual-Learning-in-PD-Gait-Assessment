import os
import glob
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from PIL import Image
from sklearn.decomposition import TruncatedSVD
import torch.nn.functional as F

# ============================================================
# 1. Symmetric variance computation engine
# ============================================================
# ============================================================
# 1. Symmetric pairwise comparison engine
# ============================================================
def compute_trimodal_analysis(feat_rgb, feat_depth, feat_ir):
    f_r, f_d, f_i = feat_rgb.detach(), feat_depth.detach(), feat_ir.detach()
    
    # --- 1. Compute variances ---
    var_r = f_r.var(dim=0).mean().item()
    var_d = f_d.var(dim=0).mean().item()
    var_i = f_i.var(dim=0).mean().item()
    
    # Symmetric variance ratio (Max / Min)
    ratio_rd = max(var_r, var_d) / min(var_r, var_d) if min(var_r, var_d) != 0 else 0
    ratio_ri = max(var_r, var_i) / min(var_r, var_i) if min(var_r, var_i) != 0 else 0
    ratio_di = max(var_d, var_i) / min(var_d, var_i) if min(var_d, var_i) != 0 else 0

    # --- 2. Compute modality gap (centroid L2 distance) ---
    # L2-normalize features onto the unit sphere before computing gap
    norm_r = F.normalize(f_r, p=2, dim=1)
    norm_d = F.normalize(f_d, p=2, dim=1)
    norm_i = F.normalize(f_i, p=2, dim=1)
    
    # Compute per-modality centroids
    centroid_r = norm_r.mean(dim=0)
    centroid_d = norm_d.mean(dim=0)
    centroid_i = norm_i.mean(dim=0)
    
    # Euclidean distance between centroids (L2 norm)
    gap_rd = torch.norm(centroid_r - centroid_d, p=2).item()
    gap_ri = torch.norm(centroid_r - centroid_i, p=2).item()
    gap_di = torch.norm(centroid_d - centroid_i, p=2).item()

    print("\n" + "🎯" + "="*60)
    print("   [Drive&Act-MIL PAIRWISE RESULTS] (Projected to 64-dim)")
    print("   " + "-"*60)
    print("   A. RAW VARIANCES:")
    print(f"      Color: {var_r:.6f} | Depth: {var_d:.6f} | IR: {var_i:.6f}")
    print("   " + "-"*60)
    print("   B. MODALITY GAP (\u0394gap) & VARIANCE RATIO (\u0394\u03c3\u00b2):")
    print(f"      1. [Color vs Depth] Gap: {gap_rd:.4f} | Var Ratio: {ratio_rd:.4f}")
    print(f"      2. [Color vs IR   ] Gap: {gap_ri:.4f} | Var Ratio: {ratio_ri:.4f}")
    print(f"      3. [Depth vs IR   ] Gap: {gap_di:.4f} | Var Ratio: {ratio_di:.4f}")
    print("   " + "="*60 + "\n")
    
    return

# ============================================================
# 2. Synchronized video frame sampling
# ============================================================
def extract_tensor_from_frame(frame, transform):
    """Convert OpenCV BGR frames to RGB tensors for ResNet"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(frame_rgb)
    return transform(img_pil).unsqueeze(0)

# ============================================================
# 3. Main entry point
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    print("\n📦 Loading Standard ResNet50...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
    resnet.fc = torch.nn.Identity() 
    
    transform_vision = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    from model.paths import DRIVE_ACT_DATA, as_str
    data_dir = as_str(DRIVE_ACT_DATA / "sub1") 
    
    # 👑 Collect .mp4 files only; ignore .timestamps
    color_videos = sorted(glob.glob(os.path.join(data_dir, '**/*color.mp4'), recursive=True))

    if not color_videos:
        print(f"❌ 在 {data_dir} 中找不到包含 'color.mp4' 的视频文件。")
        return

    print(f"🎥 Found {len(color_videos)} Color video files. Starting synchronous extraction...")

    # ==========================================
    # 👑 Sparse sampling strategy
    # ==========================================
    frame_step = 30     # Sample every 30 frames (~1s) for motion diversity
    max_samples = 100   # Stop after 100 paired samples
    
    rgb_features, depth_features, ir_features = [], [], []
    valid_count = 0

    for color_path in color_videos:
        if valid_count >= max_samples: break
        
        # Locate paired depth and IR videos
        depth_path = color_path.replace('color', 'depth')
        ir_path = color_path.replace('color', 'ir')

        if not (os.path.exists(depth_path) and os.path.exists(ir_path)):
            print(f"   ⚠️ 找不到配对的 Depth/IR 视频，跳过: {os.path.basename(color_path)}")
            continue

        # Open three videos synchronously
        cap_c = cv2.VideoCapture(color_path)
        cap_d = cv2.VideoCapture(depth_path)
        cap_i = cv2.VideoCapture(ir_path)

        frame_idx = 0
        while True:
            ret_c, frame_c = cap_c.read()
            ret_d, frame_d = cap_d.read()
            ret_i, frame_i = cap_i.read()

            # Break if any video stream ends
            if not (ret_c and ret_d and ret_i):
                break
            
            # Extract features only at frame_step intervals
            if frame_idx % frame_step == 0:
                img_c = extract_tensor_from_frame(frame_c, transform_vision).to(device)
                img_d = extract_tensor_from_frame(frame_d, transform_vision).to(device)
                img_i = extract_tensor_from_frame(frame_i, transform_vision).to(device)

                with torch.no_grad():
                    f_c = resnet(img_c).cpu()
                    f_d = resnet(img_d).cpu()
                    f_i = resnet(img_i).cpu()

                rgb_features.append(f_c)
                depth_features.append(f_d)
                ir_features.append(f_i)
                
                valid_count += 1
                if valid_count % 10 == 0 or valid_count == max_samples:
                    print(f"   ✅ [{valid_count}/{max_samples}] Sampled frame {frame_idx} from {os.path.basename(color_path)}")
                
                if valid_count >= max_samples:
                    break
            
            frame_idx += 1

        # Release video capture resources
        cap_c.release()
        cap_d.release()
        cap_i.release()

    # ==========================================
    # 3. Joint SVD reduction (fair comparison protocol)
    # ==========================================
    if valid_count > 10:
        all_rgb = torch.cat(rgb_features)
        all_depth = torch.cat(depth_features)
        all_ir = torch.cat(ir_features)
        
        joint_features = torch.cat([all_rgb, all_depth, all_ir], dim=0)
        
        fair_target_dim = 64
        print(f"\n⏳ Running Joint TruncatedSVD to project down to {fair_target_dim}-dim...")
        
        joint_features_norm = F.normalize(joint_features, p=2, dim=1)
        svd = TruncatedSVD(n_components=fair_target_dim, random_state=42)
        joint_64 = svd.fit_transform(joint_features_norm.numpy())
        
        rgb_64 = torch.tensor(joint_64[:valid_count], dtype=torch.float32)
        depth_64 = torch.tensor(joint_64[valid_count:2*valid_count], dtype=torch.float32)
        ir_64 = torch.tensor(joint_64[2*valid_count:], dtype=torch.float32)
        
        compute_trimodal_analysis(rgb_64, depth_64, ir_64)
    else:
        print("\n❌ 未能提取到足够的样本。")

if __name__ == "__main__":
    main()