import argparse
import datetime
import json
import os
import time
import numpy as np
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# FIXED: Import from timm.optim to avoid deprecation warnings
import timm.optim as optim_factory 

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

from dataloader.Jointly_Dataset import Buffer_Dataset
from medcoss_model.Unimodel import Unified_Model
from medcoss_model.Unimodel_Teacher import Teacher_Unified_Model
from engine_pretrain_er import jointly_train_one_epoch_with_teacher

# --- WEARGAIT BRIDGE ---
import sys
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from model.weargait.ewc.config import Config
from model.weargait.ewc.data_loader import build_subj2label, make_fixed_balanced_folds_no_overlap
from torch.utils.data.dataloader import default_collate
# -----------------------

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    # WearGait Parameters
    parser.add_argument('--win_len', type=int, default=120)
    parser.add_argument('--hop_len', type=int, default=60)
    parser.add_argument('--n_folds', type=int, default=10)

    # Training Hyperparameters
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--norm_pix_loss', action='store_true')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N')

    # Paths and System
    parser.add_argument('--output_dir', default='./output_dir')
    parser.add_argument('--log_dir', default='./output_dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.set_defaults(pin_mem=True)

    # Checkpointing (Required for misc.load_model)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N')
    parser.add_argument('--gpu', default=0, type=int)

    # Distributed Training
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    # Continual Learning Specifics
    parser.add_argument('--task_modality', default="1D_text", type=str)
    parser.add_argument('--load_current_pretrained_weight', default="", type=str)
    parser.add_argument('--load_teacher_weight', default="", type=str)
    parser.add_argument('--mix_up', type=int, default=1)

    return parser

def main(args):
    misc.init_distributed_mode(args)
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))

    # --- 1. WEARGAIT FOLD GENERATION ---
    def _scan_subjects(dir_path: Path):
        return sorted({x.name.split("_")[0].lower() for x in dir_path.glob(Config.CSV_PATTERN)})
        
    pd_ids = _scan_subjects(Config.PD_PATH)
    hc_ids = _scan_subjects(Config.HC_PATH)
    subj2label = build_subj2label(pd_ids, hc_ids)
    folds = make_fixed_balanced_folds_no_overlap(pd_ids, hc_ids, n_folds=args.n_folds, seed=args.seed)
    
    train_subs, test_subs = folds[0]

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # --- 2. WEARGAIT DATASET BRIDGE ---
    dataset_train = Buffer_Dataset(
        task_modality=args.task_modality,
        args=args,
        train_subs=train_subs,
        test_subs=test_subs,
        subj2label=subj2label,
        is_training=True
    )

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=default_collate
    )

    # --- 3. INITIALIZE STUDENT MODEL ---
    model = Unified_Model(now_1D_input_size=(112,1), now_2D_input_size=(512, 512), now_3D_input_size=(16, 192, 192), norm_pix_loss=args.norm_pix_loss)
    
    if args.load_current_pretrained_weight:
        print(f"Loading student weights from {args.load_current_pretrained_weight}")
        pretrained_weight = torch.load(args.load_current_pretrained_weight, map_location='cpu', weights_only=False)
        pre_dict = pretrained_weight["model"]
        model_dict = model.state_dict()
        
        update_module = ["fused_encoder", "token_embed", "cls_token", "video_embed"]
        pre_dict_update = {k: v for k, v in pre_dict.items() if (k in model_dict and any(m in k for m in update_module))}
        model_dict.update(pre_dict_update)
        model.load_state_dict(model_dict)
        print("✅ Student pre-trained model loaded.")
        del pre_dict, pretrained_weight, model_dict
        
    model.to(device)

    # --- 4. INITIALIZE TEACHER MODEL ---
    teacher_model = Teacher_Unified_Model(now_1D_input_size=(112, 1), now_2D_input_size=(512, 512), norm_pix_loss=args.norm_pix_loss)
    
    teacher_weight_path = args.load_teacher_weight if args.load_teacher_weight else args.load_current_pretrained_weight
    if teacher_weight_path:
        print(f"Loading teacher weights from {teacher_weight_path}")
        pretrained_weight = torch.load(teacher_weight_path, map_location='cpu', weights_only=False)
        teacher_model.load_state_dict(pretrained_weight["model"], strict=False)
        print("✅ Teacher pre-trained model loaded.")
        del pretrained_weight
        
    teacher_model.to(device)
    
    # Freeze Teacher
    for p in teacher_model.parameters():
        p.requires_grad = False

    # --- 5. SETUP OPTIMIZATION ---
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.gpu], find_unused_parameters=True)
        teacher_model_without_ddp = teacher_model.module
    else:
        model_without_ddp = model
        teacher_model_without_ddp = teacher_model
        
    # Updated timm call
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    
    # load_model handles resuming if args.resume is set
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # --- 6. TRAINING LOOP ---
    print(f"Start continual training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed: 
            data_loader_train.sampler.set_epoch(epoch)
            
        train_stats = jointly_train_one_epoch_with_teacher(
            model, teacher_model, teacher_model_without_ddp, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer, args=args, current_task=args.task_modality
        )
        
        if args.output_dir and ((epoch + 1) % 100 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None: log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    print(f"Training time {str(datetime.timedelta(seconds=int(time.time() - start_time)))}")

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)


# CUDA_VISIBLE_DEVICES=0 nohup python -u /home/minlin/Minlin/JBHI26/model/weargait/ewc/medcoss/main_pretrain_medcoss.py \
#     --task_modality "2D_path" \
#     --load_current_pretrained_weight "/home/minlin/Minlin/JBHI26/model/weargait/ewc/medcoss/log/stage3_walkway/checkpoint-199.pth" \
#     --mix_up 1 \
#     --batch_size 64 \
#     --epochs 200 \
#     --warmup_epochs 20 \
#     --blr 1e-3 \
#     --weight_decay 0.05 \
#     --output_dir "/home/minlin/Minlin/JBHI26/model/weargait/ewc/medcoss/log/stage3_insole/" \
#     > "/home/minlin/Minlin/JBHI26/model/weargait/ewc/medcoss/log/stage3_insole.out" 2>&1 &