import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import timm.optim as optim_factory
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from medcoss_model.Unimodel import Unified_Model
from engine_pretrain import train_one_epoch

# --- WEARGAIT BRIDGE ---
# Go 4 levels up: medcoss -> ewc -> weargait -> model -> JBHI26
project_root = Path(__file__).resolve().parents[4]
sys.path.append(str(project_root))

from model.weargait.ewc.config import Config
from model.weargait.ewc.data_loader import build_subj2label, make_fixed_balanced_folds_no_overlap
from dataloader.Jointly_Dataset import Buffer_Dataset
# -----------------------

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int)
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--norm_pix_loss', action='store_true', help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR', help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR', help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--output_dir', default='./output_dir', help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir', help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')

    # MedCoSS Modality
    parser.add_argument('--task_modality', default="1D_text", choices=["1D_text", "2D_xray", "3D_CT", "3D_MR", "2D_path"], type=str)
    parser.add_argument('--load_current_pretrained_weight', default="", type=str, help='pre-training path')

    # --- WearGait Specific Args ---
    parser.add_argument('--win_len', type=int, default=120)
    parser.add_argument('--hop_len', type=int, default=60)
    parser.add_argument('--n_folds', type=int, default=10)
    
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
    
    # Using the n_folds and seed identical to your main scripts
    folds = make_fixed_balanced_folds_no_overlap(pd_ids, hc_ids, n_folds=args.n_folds, seed=args.seed)
    
    # HARDCODED FOLD 0 FOR PIPELINE TESTING
    train_subs, test_subs = folds[0]
    print(f"Executing MedCoSS Pipeline on Fold 0 | Train: {len(train_subs)} subs, Test: {len(test_subs)} subs")
    # -----------------------------------

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
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

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
    )
    
    # --- 3. MODEL INITIALIZATION ---
    model = Unified_Model(now_1D_input_size=(112,1), now_2D_input_size=(224, 224), now_3D_input_size=(16, 192, 192), norm_pix_loss=args.norm_pix_loss)
    
    if args.load_current_pretrained_weight:
        print("load pretrained parameter from ", args.load_current_pretrained_weight)
        pretrained_weight = torch.load(args.load_current_pretrained_weight, map_location='cpu', weights_only=False)
        pre_dict = pretrained_weight["model"]
        model_dict = model.state_dict()
        
        updata_module = ["fused_encoder", "token_embed", "cls_token", "video_embed"] 
        pre_dict_update = {k: v for k, v in pre_dict.items() if (k in model_dict and sum([module in k for module in updata_module])) }
        model_dict.update(pre_dict_update)
        model.load_state_dict(model_dict)
        print("load pre-trained model success!")
        del model_dict, pretrained_weight, pre_dict
        
    model.to(device)
    model_without_ddp = model
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  
        args.lr = args.blr * eff_batch_size / 256

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
            
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
            args=args
        )
        
        if args.output_dir and ((epoch + 1) % 100 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    print('Training time {}'.format(str(datetime.timedelta(seconds=int(total_time)))))

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)