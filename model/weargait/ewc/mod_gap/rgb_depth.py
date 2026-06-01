import os
import torch
import torchvision.models as models
import torchvision.transforms as T
from moviepy import VideoFileClip
from PIL import Image
import numpy as np
from model.weargait.ewc.utility import compute_modality_analysis

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # ==========================================
    # 1. 加载模型 (ResNet50 用于提取特征, MiDaS 用于生成深度图)
    # ==========================================
    print("\n📦 Loading ResNet50 and MiDaS (Depth Estimator)...")
    
    # 我们用同一个 ResNet50 来分别提取 RGB 和 Depth 的特征 (这是 RGB-D 的常规操作)
    resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).to(device).eval()
    resnet.fc = torch.nn.Identity()
    
    # 加载 MiDaS 深度预测模型
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True).small_transform

    # 提取特征用的标准预处理
    transform_feature = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # ==========================================
    # 2. 准备数据
    # ==========================================
    video_files = []
    root_dir = "/home/minlin/ucf101_raw/UCF101_subset" 
    
    if not os.path.exists(root_dir):
        print(f"❌ 找不到文件夹 {root_dir}，请确认路径。")
        return

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith((".mp4", ".avi")):
                video_files.append(os.path.join(root, file))

    print(f"\n⚙️ Processing {len(video_files)} videos for RGB-Depth analysis...")
    
    rgb_features, depth_features = [], []
    valid_count = 0

    # ==========================================
    # 3. 提取 RGB 和 Depth 特征
    # ==========================================
    for v_path in video_files:
        if valid_count >= 50: break
        
        clip = None # 初始化
        try:
            clip = VideoFileClip(v_path)
            frame = clip.get_frame(clip.duration / 2) # 取中间帧 (numpy array)
            img_pil = Image.fromarray(frame).convert('RGB')
            
            # --- 处理 RGB ---
            img_tensor = transform_feature(img_pil).unsqueeze(0).to(device)
            
            # --- 处理 Depth ---
            # 1. 用 MiDaS 预测深度
            input_batch = midas_transforms(frame).to(device)
            with torch.no_grad():
                prediction = midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=(224, 224),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # 2. 将深度图归一化到 0-255 并转为 3 通道 (伪装成普通图片给 ResNet 吃)
            depth_map = prediction.cpu().numpy()
            depth_min, depth_max = depth_map.min(), depth_map.max()
            depth_map = 255.0 * (depth_map - depth_min) / (depth_max - depth_min)
            depth_map = depth_map.astype(np.uint8)
            
            # 复制成 3 通道
            depth_3c = np.stack((depth_map,)*3, axis=-1)
            depth_pil = Image.fromarray(depth_3c)
            depth_tensor = transform_feature(depth_pil).unsqueeze(0).to(device)

            # --- 提取特征 ---
            with torch.no_grad():
                f_rgb = resnet(img_tensor).cpu()
                f_depth = resnet(depth_tensor).cpu()

            rgb_features.append(f_rgb)
            depth_features.append(f_depth)
            valid_count += 1
            print(f"   ✅ [{valid_count}/50] Processed: {os.path.basename(v_path)}")
            
        except Exception as e:
            print(f"   ⚠️ Error processing {os.path.basename(v_path)}: {e}")
            continue
            
        finally:
            # 👑 修复：无论成功还是报错，绝对保证释放视频资源
            if clip is not None:
                clip.close()

    # ==========================================
    # 4. 计算结果
    # ==========================================
    if valid_count > 0:
        all_rgb = torch.cat(rgb_features)      # (50, 2048)
        all_depth = torch.cat(depth_features)  # (50, 2048)
        
        gaps, ratios = [], []
        
        # 👑 修复：统一降维到 64 维！与临床传感器保持绝对公平的比较基准。
        fair_target_dim = 64 
        
        for _ in range(10):
            g, r = compute_modality_analysis(all_rgb, all_depth, target_dim=fair_target_dim)
            gaps.append(g)
            ratios.append(r)
            
        avg_gap = sum(gaps) / len(gaps)
        avg_ratio = sum(ratios) / len(ratios)

        print("\n" + "🟢" + "="*45)
        print(f"   [LEVEL 1 RESULTS] RGB vs. Depth (Projected to {fair_target_dim}-dim)")
        print(f"   1. Modality Gap (\u0394gap)    : {avg_gap:.4f}")
        print(f"   2. Variance Ratio (\u0394\u03c3\u00b2) : {avg_ratio:.4f}")
        print("   " + "="*45 + "\n")
        print("💡 预期现象：")
        print("因为它们共享了极度强大的预训练 ResNet50 (自带全局 BN)，")
        print("潜特征的方差会被强行拉近。这完美证明了：即使在视觉领域，")
        print("同构特征的方差偏移也是温和的 (Ratio 接近 1)，")
        print("这也反衬出医疗异构传感器的方差断崖式下跌 (Ratio=0.4) 有多致命！")

if __name__ == "__main__":
    main()