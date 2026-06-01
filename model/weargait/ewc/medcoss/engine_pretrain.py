import math
import sys
from typing import Iterable

import torch
import util.misc as misc
import util.lr_sched as lr_sched

def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt="{global_avg:.6f}"))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir), "task_name", args.task_modality)

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        
        # --- WEARGAIT FIX: Adapt the dictionary from default_collate ---
        # Our Buffer_Dataset returns {"data": tensor, "modality": ["text", ...], "label": tensor}
        model_input = {
            "data": samples["data"].to(device, non_blocking=True),
            "modality": samples["modality"][0] # Extract the string ("text" or "2D image")
        }

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.amp.autocast('cuda'):
            # Pass directly into the Unified_Model Masked Autoencoder
            (loss, _), _, _, _ = model(model_input, mask_ratio=args.mask_ratio)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update("loss", loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update("lr", lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    global_avg_print = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged stats:", global_avg_print)
    
    return global_avg_print