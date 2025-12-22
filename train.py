"""
Training Loop and Loss Functions
Orchestrates model training with wandb logging and checkpoint management

This module provides:
- Loss functions: CB_CE_Loss, FocalLoss, fact_loss_fn
- Training orchestration with validation
- Checkpoint and model config saving
"""

import os
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed (int): Random seed value
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CB_CE_Loss(nn.Module):
    """
    Class-Balanced Cross-Entropy Loss
    
    Reference: https://ieeexplore.ieee.org/abstract/document/8953804
    
    Addresses class imbalance by re-weighting loss based on effective number of samples.
    Formula: w_i = (1 - beta) / (1 - beta^n_i)
    where n_i is the number of samples for class i.
    
    Args:
        num_samples (list/tensor): Number of samples per class
        beta (float): Hyperparameter controlling weighting strength (default 0.99)
        ignore_index (int): Class index to ignore in loss computation
    """
    
    def __init__(self, num_samples, beta=0.99, ignore_index=-100):
        super(CB_CE_Loss, self).__init__()
        
        # Convert to tensor if needed
        if not isinstance(num_samples, torch.Tensor):
            num_samples = torch.tensor(num_samples, dtype=torch.float32)
        else:
            num_samples = num_samples.float()
        
        # Calculate class-balanced weights
        effective_num = 1.0 - torch.pow(beta, num_samples)
        weights = (1.0 - beta) / (effective_num + 1e-8)
        self.register_buffer('weights', weights)
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model outputs, shape (batch_size, num_classes)
            targets (torch.Tensor): Target indices, shape (batch_size,)
            
        Returns:
            torch.Tensor: Loss vector, shape (batch_size,) with reduction='none'
        """
        if targets.dim() > 1:
            targets = targets.argmax(dim=-1)
        
        # Standard cross entropy loss (per sample)
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        
        # Apply class weights
        class_weights = self.weights.to(logits.device)
        
        # Safe lookup: for ignore_index, use dummy index to avoid out-of-bounds
        safe_targets = targets.clone()
        safe_targets[safe_targets == self.ignore_index] = 0
        sample_weights = class_weights[safe_targets]
        
        # Weight the loss
        weighted_loss = ce_loss * sample_weights
        
        # Mask out ignored indices
        weighted_loss = torch.where(
            targets != self.ignore_index, 
            weighted_loss, 
            torch.zeros_like(weighted_loss)
        )
        
        return weighted_loss  # Returns [batch_size]


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    
    Reference: https://arxiv.org/abs/1708.02002
    
    Reduces weight of easy negative examples and focuses training on hard examples.
    Formula: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    Args:
        alpha (float/list): Weighting factor for each class (default 0.25)
        gamma (float): Focusing parameter (default 2.0)
        reduction (str): 'mean', 'sum', or 'none'
        ignore_index (int): Class index to ignore
    """
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction="none", ignore_index=-100):
        super(FocalLoss, self).__init__()
        
        # Prepare alpha weights
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([1 - alpha, alpha])  # [alpha_neg, alpha_pos]
        else:
            self.alpha = torch.tensor(alpha)
        
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        """
        Args:
            logits (torch.Tensor): Model outputs, shape (batch_size, num_classes)
            targets (torch.Tensor): Target indices, shape (batch_size,)
            
        Returns:
            torch.Tensor: Loss (scalar if reduction='mean', vector if reduction='none')
        """
        device = logits.device
        self.alpha = self.alpha.to(device)
        
        if targets.dim() > 1:
            targets = targets.argmax(dim=-1)
        
        # Compute base cross-entropy loss
        ce_loss = self.ce_loss(logits, targets)  # (batch_size,)
        p_t = torch.exp(-ce_loss)  # Probability of correct class
        
        # Focal term: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Alpha weight (balanced loss)
        safe_targets = targets.clone()
        safe_targets[safe_targets == self.ignore_index] = 0
        alpha_weight = self.alpha.gather(0, safe_targets.view(-1))
        
        # Compute focal loss
        focal_loss = alpha_weight * focal_weight * ce_loss
        
        # Mask out ignored indices
        focal_loss = torch.where(
            targets != self.ignore_index,
            focal_loss,
            torch.zeros_like(focal_loss)
        )

        # Apply reduction
        if self.reduction == "mean":
            valid_mask = targets != self.ignore_index
            return focal_loss.sum() / valid_mask.sum() if valid_mask.any() else torch.tensor(0.0, device=device)
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss  # Returns [batch_size]


def hits_at_k(predictions, targets, k=5):
    """
    Calculate Hits@K metric for classification.
    
    Measures if the correct class is in the top-K predicted classes.
    
    Args:
        predictions (torch.Tensor): Prediction logits, shape (batch_size, num_classes)
        targets (torch.Tensor): Target class indices, shape (batch_size,)
        k (int): Number of top predictions to consider
        
    Returns:
        float: Hits@K score (0-1)
    """
    # Filter valid samples
    valid_mask = targets != -100
    targets = targets[valid_mask]
    predictions = predictions[valid_mask]
    
    if targets.numel() == 0:
        return 0.0
    
    # Get top-k predictions
    top_k_preds = torch.topk(predictions, k, dim=-1).indices  # (batch_size, k)
    
    # Convert target indices to one-hot if needed
    if targets.dim() == 1:
        targets = torch.nn.functional.one_hot(targets, num_classes=predictions.size(1))
    
    targets = targets.float()
    
    # Check if any top-k prediction matches the target
    hits = torch.any(targets.gather(1, top_k_preds), dim=1).float()
    
    return hits.mean().item()


def compute_loss(outputs, targets, values, task_weights=None, gate_reg_weights=None, hits_k=False):
    """
    Compute multi-task loss with gate-based regularization.
    
    Architecture:
    - Each task has a raw loss (tag, time, scale, negative)
    - Each task has a learnable gate that modulates importance
    - Final loss: L_final = (1 - g_task) * L_raw + lambda_task * g_task^2
    
    The gate values (0-1) represent sample importance:
    - g = 0: Sample is clean, use full loss
    - g = 1: Sample is noisy, mostly ignore
    
    Args:
        outputs (dict): Model outputs with keys: tag, time, scale, negative, gates
        targets (dict): Target tensors with keys: tag, time, scale, negative
        values (torch.Tensor): Numeric values (future use for fact regression)
        task_weights (dict): Loss weights for each task (default all 1.0)
        gate_reg_weights (dict): Gate regularization lambdas (default all 1.0)
        hits_k (bool): Whether to compute Hits@K metrics
        
    Returns:
        tuple: (total_loss, losses_log) or (total_loss, losses_log, hits_k_dict)
    """
    # Default values
    if task_weights is None:
        task_weights = {"tag": 1.0, "time": 1.0, "scale": 1.0, "negative": 1.0}
    if gate_reg_weights is None:
        gate_reg_weights = {"tag": 1.0, "time": 1.0, "scale": 1.0, "negative": 1.0}
    
    losses = {}
    
    # ===== Compute Raw Loss Vectors [batch_size] =====
    # Each loss function should return shape [batch_size] with reduction='none'
    
    if "tag" in targets:
        # Placeholder: assumes tag_loss_fn is defined in training script
        # For now, use cross entropy
        losses["tag"] = F.cross_entropy(
            outputs["tag"], 
            targets["tag"], 
            reduction='none',
            ignore_index=-100
        )

    if "time" in targets:
        losses["time"] = F.cross_entropy(
            outputs["time"],
            targets["time"],
            reduction='none',
            ignore_index=-100
        )
    
    if "scale" in targets:
        losses["scale"] = F.cross_entropy(
            outputs["scale"],
            targets["scale"],
            reduction='none',
            ignore_index=-100
        )
    
    if "negative" in targets:
        losses["negative"] = F.cross_entropy(
            outputs["negative"],
            targets["negative"],
            reduction='none',
            ignore_index=-100
        )
    
    # ===== Extract Gates =====
    gates = outputs.get("gates", {})
    
    # Initialize total loss vector
    device = list(losses.values())[0].device
    total_loss_vec = torch.zeros(list(losses.values())[0].shape[0], device=device)
    
    # ===== Apply Gate-based Weighting =====
    # Formula: L_final = (1 - g_task) * L_raw + lambda_task * g_task^2
    
    for task_name, raw_loss in losses.items():
        gate = gates[task_name].squeeze(-1)  # (batch_size, 1) -> (batch_size,)
        lambda_task = gate_reg_weights.get(task_name, 1.0)
        task_weight = task_weights.get(task_name, 1.0)
        
        # Gate-weighted loss
        gated_loss = (1 - gate) * raw_loss + lambda_task * gate.pow(2)
        
        # Add task-weighted contribution
        total_loss_vec += task_weight * gated_loss
    
    # ===== Final Average Loss =====
    total_loss = total_loss_vec.mean()
    
    # ===== Prepare Logging Dict =====
    losses_log = {k: v.mean().item() for k, v in losses.items()}
    
    # Log gate statistics
    for task_name, gate in gates.items():
        gate_mean = gate.mean().item()
        gate_std = gate.std().item() if gate.numel() > 1 else 0.0
        losses_log[f'gate_{task_name}_mean'] = gate_mean
        losses_log[f'gate_{task_name}_std'] = gate_std
        losses_log[f'lambda_{task_name}'] = gate_reg_weights.get(task_name, 1.0)
    
    # ===== Compute Hits@K (Optional) =====
    hits_at_k_dict = {}
    if hits_k and "tag" in targets:
        hits_at_k_dict["hits_1"] = hits_at_k(outputs["tag"], targets["tag"], k=1)
        hits_at_k_dict["hits_3"] = hits_at_k(outputs["tag"], targets["tag"], k=3)
        hits_at_k_dict["hits_5"] = hits_at_k(outputs["tag"], targets["tag"], k=5)

    return (total_loss, losses_log, hits_at_k_dict) if hits_k else (total_loss, losses_log)


def save_checkpoint(model, optimizer, scheduler, epoch, step, save_path):
    """
    Save training checkpoint for resuming training.
    
    Args:
        model (nn.Module): Model to save
        optimizer (Optimizer): Optimizer state
        scheduler (LRScheduler): Learning rate scheduler state
        epoch (int): Current epoch
        step (int): Current training step
        save_path (str): Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }
    torch.save(checkpoint, save_path)
    print(f"✅ Checkpoint saved at {save_path}")


def load_checkpoint(model, optimizer, scheduler, save_path, device):
    """
    Load training checkpoint to resume training.
    
    Args:
        model (nn.Module): Model to load into
        optimizer (Optimizer): Optimizer to load state into
        scheduler (LRScheduler): Scheduler to load state into
        save_path (str): Path to checkpoint
        device (str): Device to load to
        
    Returns:
        tuple: (epoch, step)
    """
    checkpoint = torch.load(save_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    print(f"✅ Checkpoint loaded: epoch {epoch}, step {step}")
    return epoch, step


def save_model_config(model, tag2id, time2id, scale2id, num_tags, num_times, num_scales,
                      output_dir, run_name):
    """
    Save model configuration and ID mappings for later loading.
    
    Essential for ensuring the test/evaluation uses the exact same mappings
    as the training run.
    
    Args:
        model (nn.Module): Trained model (for extracting architecture info)
        tag2id (dict): Tag name to index mapping
        time2id (dict): Time description to index mapping
        scale2id (dict): Scale to index mapping
        num_tags (int): Total number of tag classes
        num_times (int): Total number of time classes
        num_scales (int): Total number of scale classes
        output_dir (str): Directory to save config
        run_name (str): Run identifier for config filename
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Reverse mappings for id-to-label
    id2tag = {str(idx): tag for tag, idx in tag2id.items()}
    id2time = {str(idx): time for time, idx in time2id.items()}
    id2scale = {str(idx): scale for scale, idx in scale2id.items()}
    
    # Compile configuration
    config = {
        "run_name": run_name,
        "bert_model": "nlpaueb/sec-bert-base",
        
        # Mappings
        "tag2id": tag2id,
        "id2tag": id2tag,
        "time2id": time2id,
        "id2time": id2time,
        "scale2id": scale2id,
        "id2scale": id2scale,
        
        # Dimensions
        "num_tags": num_tags,
        "num_times": num_times,
        "num_scales": num_scales,
        
        # Constants
        "classification_missing_value": -100,
        "numeric_missing_value": float(torch.finfo(torch.float32).max),
        
        # Model architecture parameters
        "model_args": {
            "bert_model_name": "nlpaueb/sec-bert-base",
            "num_tags": num_tags,
            "num_times": num_times,
            "num_scales": num_scales
        }
    }
    
    # Save to JSON
    config_path = os.path.join(output_dir, f"{run_name}_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Configuration saved to: {config_path}")


def validate_model(model, val_loader, task_weights, gate_reg_weights, device):
    """
    Run validation loop to compute validation metrics.
    
    Args:
        model (nn.Module): Model to validate
        val_loader (DataLoader): Validation data loader
        task_weights (dict): Task loss weights
        gate_reg_weights (dict): Gate regularization weights
        device (str): Device to run on
        
    Returns:
        tuple: (avg_loss, loss_dict) where loss_dict contains per-task and gate stats
    """
    model.eval()
    val_loss = 0
    valid_batch_count = 0
    all_losses = {
        'tag': 0, 'time': 0, 'scale': 0, 'negative': 0,
        'gate_tag_mean': 0, 'gate_time_mean': 0, 'gate_scale_mean': 0, 'gate_negative_mean': 0
    }
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_tokens = batch["start_token"].to(device)
            end_tokens = batch["end_token"].to(device)
            values = batch.get("value", torch.zeros(len(batch))).to(device)
            
            targets = {
                "tag": batch["tag"].to(device),
                "time": batch["time"].to(device),
                "scale": batch["scale"].to(device),
                "negative": batch["negative"].to(device)
            }
            
            outputs = model(input_ids, attention_mask, start_tokens, end_tokens)
            loss, losses = compute_loss(outputs, targets, values, task_weights, gate_reg_weights)
            
            val_loss += loss.item()
            valid_batch_count += 1
            
            for key in all_losses:
                if key in losses:
                    all_losses[key] += losses[key]

    # Average loss
    val_loss /= valid_batch_count
    for key in all_losses:
        all_losses[key] /= valid_batch_count
    
    return val_loss, all_losses
