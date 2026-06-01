import math
import torch
import torch.nn as nn
import numpy as np
import ot
from model.baselines.LwI.groundmetric import GroundMetric

def get_histogram(args, idx, cardinality, layer_name):
    """Returns uniform histogram for OT unless importance is specified."""
    if not args.unbalanced:
        return np.ones(cardinality) / cardinality
    else:
        return np.ones(cardinality)

def get_wassersteinized_layers_modularized(args, device, networks, ignore_keyword='encoders', eps=1e-7):
    """
    Modified LwI Fusion for WearGait.
    
    Logic:
    1. Skip any layer containing 'ignore_keyword' (Private Encoders).
    2. Fuse Shared Layers (Backbone + Head).
    3. Apply Max Similarity to early shared layers (Alignment).
    4. Apply Min Similarity to the last 'args.layers' shared layers (Pathway Protection).
    
    Returns:
        fused_weights (dict): {layer_name: fused_tensor}
    """
    ground_metric_object = GroundMetric(args)
    
    # 1. Identify Fusable Layers (Shared Backbone + Head)
    model_new = networks[1]
    model_old = networks[0]
    
    fusable_layers = []
    for name, p in model_new.named_parameters():
        if ignore_keyword in name:
            continue
        if 'num_batches_tracked' in name: # Skip BN stats
            continue
        fusable_layers.append(name)
        
    num_fusable = len(fusable_layers)
    print(f"[OT Fusion] Found {num_fusable} shared layers to fuse (Skipped '{ignore_keyword}').")

    # 2. Iterate and Fuse
    fused_weights = {}
    T_var = None # Transport map state
    
    params_old = dict(model_old.named_parameters())
    params_new = dict(model_new.named_parameters())

    for idx, layer_name in enumerate(fusable_layers):
        w_old = params_old[layer_name]
        w_new = params_new[layer_name]
        
        # --- Sanity Check ---
        if w_old.shape != w_new.shape:
            print(f"  [Warning] Shape mismatch at {layer_name}. Skipping fusion.")
            fused_weights[layer_name] = w_new.data.clone()
            continue

        # --- Determine Logic: Shallow (Max Sim) vs Deep (Min Sim) ---
        if idx >= (num_fusable - args.layers):
            # Deep Layer -> Divert (Min Similarity)
            use_min_sim = True
            step = args.ensemble_step_diff 
            mode_str = "MIN Sim (Protect)"
        else:
            # Shallow Layer -> Align (Max Similarity)
            use_min_sim = False
            step = args.ensemble_step      
            mode_str = "MAX Sim (Align)"

        # --- Prepare Data ---
        layer_shape = w_old.shape
        is_conv = len(layer_shape) > 1 
        
        # Flatten: (Out_Channels, In_Features_Flatted)
        if is_conv:
            w_old_flat = w_old.data.view(layer_shape[0], -1)
            w_new_flat = w_new.data.view(layer_shape[0], -1)
        else:
            # --- FIX FOR INDEX ERROR ---
            # If weight is 1D (e.g., Bias), unsqueeze to make it (N, 1)
            if w_old.data.dim() == 1:
                w_old_flat = w_old.data.unsqueeze(1)
                w_new_flat = w_new.data.unsqueeze(1)
            else:
                w_old_flat = w_old.data
                w_new_flat = w_new.data

        # --- 3. Calculate Distance Matrix M ---
        # Calculate Ground Metric (Distance)
        M = ground_metric_object.process(w_old_flat, w_new_flat)
        cpuM = M.data.cpu().numpy()

        # --- 4. Apply LwI Logic (Invert M for Deep Layers) ---
        if use_min_sim:
            cpuM = -cpuM 

        # --- 5. Solve OT (Sinkhorn or EMD) ---
        mu = get_histogram(args, 0, layer_shape[0], layer_name)
        nu = get_histogram(args, 1, layer_shape[0], layer_name)
        
        # Calculate Permutation Matrix T
        T = ot.emd(mu, nu, cpuM) # Exact OT
        T_var = torch.from_numpy(T).to(device).float()

        # --- 6. Fuse Weights ---
        # Map Old Weights -> New Space using T
        # For bias (1D) or Linear (2D), the multiplication logic is similar for dim 0
        if w_old_flat.dim() == 2:
             # Standard multiplication
             w_old_transported = torch.matmul(T_var.t(), w_old_flat).view(layer_shape)
        else:
             # Fallback (should be covered by the unsqueeze above)
             w_old_transported = torch.matmul(T_var.t(), w_old.data)

        # Geometric Averaging
        w_fused = (1 - step) * w_old_transported + step * w_new.data
        
        fused_weights[layer_name] = w_fused
        
        print(f"  Fused {layer_name} [{mode_str}]: step={step}")

    return fused_weights