import argparse
import os
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from sklearn.cluster import KMeans
import tqdm
import sys
from pathlib import Path

# --- WEARGAIT BRIDGE ---
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from model.weargait.ewc.config import Config
from model.weargait.ewc.data_loader import build_subj2label, make_fixed_balanced_folds_no_overlap
from dataloader.Jointly_Dataset import Buffer_Dataset
from torch.utils.data.dataloader import default_collate

# Fixed Namespace Collision
from medcoss_model.Unimodel import Unified_Model
import util.misc as misc
# -----------------------

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--win_len', type=int, default=120)
    parser.add_argument('--hop_len', type=int, default=60)
    parser.add_argument('--n_folds', type=int, default=10)

    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=False)

    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--task_modality', default="1D_text", choices=["1D_text", "2D_xray", "3D_CT", "3D_MR", "2D_path"], type=str)
    parser.add_argument('--load_current_pretrained_weight', default="", type=str)
    parser.add_argument('--num_center', type=float, required=True)
    parser.add_argument('--buffer_ratio', type=float, required=True)
    parser.add_argument('--exp_name', type=str, required=True)
    return parser

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def estimate_kmean(save_path, model, data_loader, device, args):
    print(f"Executing K-Means for modality: {args.task_modality}")
    img_feature = []
    sample_indices = []
    current_idx = 0

    # 1. Extract Latent Features
    for samples in tqdm.tqdm(data_loader):
        batch_size = samples["data"].size(0)
        
        # Format dictionary exactly as Unimodel expects
        model_input = {
            "data": samples["data"].to(device, non_blocking=True),
            "modality": samples["modality"][0] # PyTorch collate makes this a tuple, grab the string
        }

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # Extract features from the encoder
                feature, _ = model(model_input, mask_ratio=0.0, feature=True)
                feature = feature.mean(1) # Average pool the tokens

        img_feature.append(feature.cpu().detach().numpy())
        
        # Track the absolute dataset index for this batch
        batch_indices = list(range(current_idx, current_idx + batch_size))
        sample_indices.extend(batch_indices)
        current_idx += batch_size

    img_feature = np.concatenate(img_feature, axis=0)
    sample_indices = np.array(sample_indices)
    
    # 2. Compute Clusters
    total_buffer_size = int(img_feature.shape[0] * args.buffer_ratio)
    if total_buffer_size < 1: total_buffer_size = 1

    # Handle both absolute numbers and fractions dynamically
    if args.num_center > 1:
        num_clusters = int(args.num_center)
    else:
        num_clusters = int(img_feature.shape[0] * args.num_center)

    # SAFETY CLAMP: You cannot have more clusters than your total buffer size or dataset size
    num_clusters = min(num_clusters, total_buffer_size, img_feature.shape[0])
    if num_clusters < 1: num_clusters = 1

    sample_num_each_clusters = total_buffer_size // num_clusters
    if sample_num_each_clusters < 1: sample_num_each_clusters = 1
        
    print(f"Total samples: {img_feature.shape[0]} | Target Buffer Size: {total_buffer_size}")
    print(f"Executing K-Means with {num_clusters} centers | Samples per center: {sample_num_each_clusters}")

    kmeans = KMeans(n_clusters=num_clusters, n_init='auto')
    kmeans.fit(img_feature)

    distances_to_cluster_centers = np.linalg.norm(img_feature - kmeans.cluster_centers_[kmeans.labels_], axis=1)

    # 3. Extract Top K Indices per Center
    buffer_indices = []
    for i in range(kmeans.n_clusters):
        cluster_distances = distances_to_cluster_centers[kmeans.labels_ == i]
        cluster_dataset_indices = sample_indices[kmeans.labels_ == i]
        
        # Get indices of the smallest distances in this cluster
        top_k_local_indices = cluster_distances.argsort()[:sample_num_each_clusters]
        
        # Map local cluster indices back to absolute dataset indices
        top_k_absolute_indices = cluster_dataset_indices[top_k_local_indices].tolist()
        buffer_indices.extend(top_k_absolute_indices)

    # 4. Save the Buffer Indices
    file_path = f'{args.task_modality}_buffer_indices.json'
    full_save_path = os.path.join(save_path, file_path)
    
    if os.path.exists(full_save_path):
        os.remove(full_save_path)
        
    save_json({"buffer_indices": buffer_indices}, full_save_path)
    print(f"Successfully saved {len(buffer_indices)} representative samples to {file_path}")

def main(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    def _scan_subjects(dir_path: Path):
        return sorted({x.name.split("_")[0].lower() for x in dir_path.glob(Config.CSV_PATTERN)})
        
    pd_ids = _scan_subjects(Config.PD_PATH)
    hc_ids = _scan_subjects(Config.HC_PATH)
    subj2label = build_subj2label(pd_ids, hc_ids)
    folds = make_fixed_balanced_folds_no_overlap(pd_ids, hc_ids, n_folds=args.n_folds, seed=args.seed)
    
    train_subs, test_subs = folds[0]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
  
    dataset_train = Buffer_Dataset(
        task_modality=args.task_modality,
        args=args,
        train_subs=train_subs,
        test_subs=test_subs,
        subj2label=subj2label,
        is_training=True  
    )

    model = Unified_Model(now_1D_input_size=(112, 1), now_2D_input_size=(512, 512), now_3D_input_size=(16, 192, 192), norm_pix_loss=args.norm_pix_loss)
    print("Loading pretrained parameter from ", args.load_current_pretrained_weight)
    pretrained_weight = torch.load(args.load_current_pretrained_weight, map_location='cpu', weights_only=False)
    model.load_state_dict(pretrained_weight["model"], strict=False)
    model.to(device)
    model.eval() # MUST be in eval mode for feature extraction

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=128,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False, 
        collate_fn=default_collate
    )
    
    # Safely extract the save path directory
    save_path = args.load_current_pretrained_weight.rsplit("/", 1)[0]
    estimate_kmean(save_path, model, data_loader_train, device, args)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


# nohup python -u /home/minlin/Minlin/JBHI26/model/weargait/ewc/medcoss/main_buffer_kmean.py \
#   --task_modality "1D_text" \
#   --load_current_pretrained_weight "/home/minlin/Minlin/JBHI26/model/weargait/ewc/medcoss/log/checkpoint-199.pth" \
#   --num_center 10 \
#   --buffer_ratio 0.1 \
#   --exp_name "imu_buffer" \
#   --output_dir "/home/minlin/Minlin/JBHI26/model/weargait/ewc/medcoss/log/" \
#   > "/home/minlin/Minlin/JBHI26/model/weargait/ewc/medcoss/log/stage2_kmean.out" 2>&1 &