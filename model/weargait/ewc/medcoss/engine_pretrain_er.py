import math
import sys
from typing import Iterable
import torch
import util.misc as misc
import util.lr_sched as lr_sched

def jointly_train_one_epoch_with_teacher(model: torch.nn.Module, teacher_model, teacher_model_without_ddp,
                            data_loader: Iterable, optimizer: torch.optim.Optimizer,
                            device: torch.device, epoch: int, loss_scaler,
                            log_writer=None, args=None, current_task=None):
    
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt="{global_avg:.6f}"))
    metric_logger.add_meter('mse_distill', misc.SmoothedValue(window_size=1))
    metric_logger.add_meter('mae_current', misc.SmoothedValue(window_size=1))

    header = f'Epoch: [{epoch}]'
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    # Map the command-line task string to the internal Unimodel forward string
    mod_map = {'1D_text': 'text', '2D_xray': '2D image', '2D_path': '2D image'}
    current_forward_modality = mod_map[args.task_modality]

    for data_iter_step, samples in enumerate(metric_logger.log_every(data_loader, 100, header)):
        
        # --- WEARGAIT FIX: Adapt dictionary from default_collate ---
        now_task_modality = samples["modality"][0] # "text" or "2D image"
        
        model_input = {
            "data": samples["data"].to(device, non_blocking=True),
            "modality": now_task_modality
        }

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        with torch.amp.autocast('cuda'):
            # Check if this batch is pulled from the Rehearsal Buffer (Past Task)
            is_past_task = (now_task_modality != current_forward_modality)

            if is_past_task:
                # --- INTRA-MODAL MIXUP (IMM) ---
                if args.mix_up == 1:
                    N = model_input["data"].size(0)
                    perm = torch.randperm(N)
                    data_shuffled = model_input["data"][perm]
                    
                    # Generate continuous lambda broadcasting shape based on dimensionality
                    if now_task_modality == "text": 
                        # IMU Data: [N, Channels, Length] -> [N, 1, 1]
                        lambda_value = torch.rand(N, 1, 1).cuda()
                    elif now_task_modality == "2D image":
                        # Spatial Data: [N, Channels, H, W] -> [N, 1, 1, 1]
                        lambda_value = torch.rand(N, 1, 1, 1).cuda()
                    else:
                        lambda_value = torch.rand(N, 1).cuda()
                        
                    # Mathematically safe continuous interpolation for WearGait sensors
                    model_input["data"] = lambda_value * model_input["data"] + (1 - lambda_value) * data_shuffled
                        
                # --- FEATURE DISTILLATION ---
                if model_input['data'].dim() == 4 and model_input['data'].size(2) == 78 and model_input['data'].size(1) != 1:
                    model_input['data'] = model_input['data'][:, 0:1, :, :]
                latent_out, noise = model(model_input.copy(), mask_ratio=args.mask_ratio, feature=True)
                
                with torch.no_grad():
                    # Teacher MUST receive the random noise mask used by Student
                    target_out = teacher_model(model_input.copy(), mask_ratio=args.mask_ratio, feature=True, noise=noise)
                
                loss = ((target_out.detach() - latent_out) ** 2).mean()
                metric_logger.update("mse_distill", loss.item())
                
            else:
                # --- CURRENT TASK MAE PRE-TRAINING ---
                (loss, _), _, _, _ = model(model_input.copy(), mask_ratio=args.mask_ratio)
                metric_logger.update("mae_current", loss.item())
              
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(), update_grad=(data_iter_step + 1) % accum_iter == 0)
        
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update("lr", lr)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', misc.all_reduce_mean(loss_value), epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    metric_logger.synchronize_between_processes()
    global_avg_print = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged stats:", global_avg_print)
    
    return global_avg_print

