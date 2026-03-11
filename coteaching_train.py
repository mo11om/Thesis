"""
Co-teaching Training Script for Multi-Task XBRL Model
=====================================================

Based on: "Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels"

This script trains two identical BERT models using the co-teaching paradigm:
- Model 1 selects small-loss samples to train Model 2, and vice versa.
- A forget rate schedule gradually increases the fraction of discarded samples.

Usage:
    python coteaching_train.py
"""

import os
import json
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from secbert_utils import (
    set_seed,
    MultiTaskModel,
    ProcessedIterableDataset,
    BufferedShuffleDataset,
    compute_coteaching_loss,
    compute_loss,
    save_coteaching_checkpoint,
    load_coteaching_checkpoint,
    CB_CE_Loss,
    FocalLoss,
    CLASSIFICATION_MISSING_VALUE,
)

# ============================================================================
# Configuration
# ============================================================================

config = {
    # --- Model ---
    "bert_model_name": "nlpaueb/sec-bert-base",
    "model_name": "secbert_coteaching",

    # --- Data ---
    "train_files": ["processed_iterable_dataset/train_400k.jsonl"],
    "valid_files": ["processed_iterable_dataset/valid_50k.jsonl"],
    "batch_size": 32,
    "num_workers": 4,
    "buffer_size": 8000,

    # --- Training ---
    "num_epochs": 30,
    "eval_step": 20000,
    "patience": 5,
    "max_saved_models": 2,
    "seed": 42,
    
    # --- Learning Rates ---
    "bert_lr": 1e-5,
    "tag_head_lr": 5e-4,
    "time_head_lr": 1e-5,
    "scale_head_lr": 3e-5,
    "negative_head_lr": 2e-5,
    "weight_decay": 1e-2,
    "eta_min": 1e-7,

    # --- Task Weights ---
    "task_weights": {"tag": 10.0, "time": 0.3, "scale": 0.2, "negative": 10.0},

    # --- Co-teaching ---
    "noise_rate": 0.1,       # 估計的雜訊率 (forget_rate 的上限)
    "num_gradual": 1,       # 預熱 epochs，forget_rate 在此期間線性增長

    # --- wandb ---
    "wandb_project": "multi-task-model",
    "date": "0213",
    "index": 1,
}

# ============================================================================
# Forget Rate Schedule
# ============================================================================

def get_forget_rate(epoch, noise_rate, num_gradual):
    """
    Co-teaching forget rate schedule.
    
    forget_rate = min(noise_rate * (epoch / num_gradual + 1), noise_rate)
    
    During warmup (epoch < num_gradual), the rate linearly increases.
    After warmup, it stays at noise_rate.
    """
    forget_rate = noise_rate * min((epoch + 1) / num_gradual, 1.0)
    return forget_rate

# ============================================================================
# Validation
# ============================================================================

def validate_coteaching(model, val_loader, task_weights, device,
                        tag_loss_fn=None, time_loss_fn=None,
                        scale_loss_fn=None, neg_loss_fn=None):
    """驗證模型 (使用單一模型進行驗證)"""
    model.eval()
    val_loss = 0
    valid_batch_count = 0
    all_losses = {'tag': 0, 'time': 0, 'scale': 0, 'negative': 0}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_tokens = batch["start_token"].to(device)
            end_tokens = batch["end_token"].to(device)
            values = batch["value"].to(device)

            targets = {
                "tag": batch["tag"].to(device),
                "time": batch["time"].to(device),
                "scale": batch["scale"].to(device),
                "negative": batch["negative"].to(device),
            }

            outputs = model(input_ids, attention_mask, start_tokens, end_tokens)
            loss, losses = compute_loss(
                outputs, targets, values, task_weights,
                tag_loss_fn=tag_loss_fn,
                time_loss_fn=time_loss_fn,
                scale_loss_fn=scale_loss_fn,
                neg_loss_fn=neg_loss_fn,
            )

            val_loss += loss.item()
            valid_batch_count += 1

            for key in all_losses:
                loss_value = losses.get(key, 0)
                all_losses[key] += loss_value if isinstance(loss_value, (int, float)) else loss_value.item()

    val_loss /= max(valid_batch_count, 1)
    for key in all_losses:
        all_losses[key] /= max(valid_batch_count, 1)

    return val_loss, all_losses

# ============================================================================
# Main Training
# ============================================================================

def main():
    cfg = config
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Data Setup (must match the notebook's preprocessing) ---
    # Tag list: load from the same counter file used during data preprocessing,
    # apply the same rare-tag filtering (threshold=10) as the notebook.
    tag_count_path = '../processed_data_task1_smaller/counter/tag_count_train_400k.json'
    with open(tag_count_path, 'r', encoding='utf-8') as f:
        tag_counter = json.load(f)
    
    # Sort by count descending (same as notebook)
    tag_counter = dict(sorted(tag_counter.items(), key=lambda item: item[1], reverse=True))
    count_threshold = 10
    standard_rare_tags = {tag for tag, count in tag_counter.items() if count < count_threshold}
    tag_list = [tag for tag in tag_counter.keys() if tag not in standard_rare_tags]
    tag2id = {tag: idx for idx, tag in enumerate(tag_list)}
    print(f"Tag list: {len(tag_list)} tags (filtered {len(standard_rare_tags)} rare tags)")

    # Time list: 9 categories (must match notebook)
    time_list = [
        'instant; past', 'instant; current', 'instant; future',
        'period; past', 'period; current', 'period; future',
        'period; past_current', 'period; current_future', 'period; past_future',
    ]
    time2id = {time: idx for idx, time in enumerate(time_list)}

    scale_list = [str(i) for i in range(-12, 13)]
    scale2id = {scale: idx for idx, scale in enumerate(scale_list)}

    print(f"Classes: tag={len(tag_list)}, time={len(time_list)}, scale={len(scale_list)}, negative=2")

    # DataLoaders
    train_dataset = ProcessedIterableDataset(cfg["train_files"])
    train_dataset = BufferedShuffleDataset(train_dataset, buffer_size=cfg["buffer_size"])
    train_loader = DataLoader(
        train_dataset, batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], drop_last=True,
    )

    valid_dataset = ProcessedIterableDataset(cfg["valid_files"])
    valid_loader = DataLoader(
        valid_dataset, batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"], drop_last=False,
    )

    # Approximate batch counts (for progress bar and scheduling)
    train_total_samples = 400000  # approximate
    train_approx_batches = math.ceil(train_total_samples / cfg["batch_size"])

    # --- Loss Functions for validation ---
    num_tag_samples = [tag_counter.get(tag, 1) for tag in tag_list]
    tag_loss_fn = CB_CE_Loss(num_tag_samples, beta=0.99)

    # Time loss: use CB_CE_Loss with counts from counter file if available
    time_count_path = 'processed_iterable_dataset/counter/secbert_train_small_time.json'
    if os.path.exists(time_count_path):
        with open(time_count_path, 'r') as f:
            time_counter = json.load(f)
        num_time_samples = [time_counter.get(t, 1) for t in time_list]
    else:
        num_time_samples = [1] * len(time_list)
    time_loss_fn = CB_CE_Loss(num_time_samples)

    scale_count_path = 'processed_iterable_dataset/counter/secbert_train_small_scale.json'
    with open(scale_count_path, 'r') as f:
        scale_counter = json.load(f)
    num_scale_samples = [scale_counter.get(s, 1) for s in scale_list]
    scale_class_weights = torch.tensor(
        [1.0 / max(n, 1) for n in num_scale_samples], dtype=torch.float
    )
    scale_class_weights = scale_class_weights / scale_class_weights.sum() * len(scale_list)
    scale_class_weights = scale_class_weights.to(device)
    scale_loss_fn = nn.CrossEntropyLoss(
        weight=scale_class_weights, ignore_index=CLASSIFICATION_MISSING_VALUE
    )

    neg_loss_fn = FocalLoss(alpha=0.25, gamma=3.0, reduction="mean")

    # --- Dual Model Initialization ---
    model1 = MultiTaskModel(
        cfg["bert_model_name"],
        num_tags=len(tag_list),
        num_times=len(time_list),
        num_scales=len(scale_list),
    ).to(device)

    model2 = MultiTaskModel(
        cfg["bert_model_name"],
        num_tags=len(tag_list),
        num_times=len(time_list),
        num_scales=len(scale_list),
    ).to(device)

    print(f"Model 1 and Model 2 initialized ({cfg['bert_model_name']})")

    # --- Dual Optimizers ---
    def make_optimizer(model):
        return AdamW([
            {"params": model.bert.parameters(), "lr": cfg["bert_lr"], "weight_decay": cfg["weight_decay"]},
            {"params": model.tag_head.parameters(), "lr": cfg["tag_head_lr"], "weight_decay": cfg["weight_decay"]},
            {"params": model.time_head.parameters(), "lr": cfg["time_head_lr"], "weight_decay": cfg["weight_decay"]},
            {"params": model.scale_head.parameters(), "lr": cfg["scale_head_lr"], "weight_decay": cfg["weight_decay"]},
            {"params": model.negative_head.parameters(), "lr": cfg["negative_head_lr"], "weight_decay": cfg["weight_decay"]},
        ])

    optimizer1 = make_optimizer(model1)
    optimizer2 = make_optimizer(model2)

    num_total_steps = cfg["num_epochs"] * train_approx_batches
    scheduler1 = CosineAnnealingLR(optimizer1, T_max=num_total_steps // 4, eta_min=cfg["eta_min"])
    scheduler2 = CosineAnnealingLR(optimizer2, T_max=num_total_steps // 4, eta_min=cfg["eta_min"])

    # --- wandb ---
    run_name = f"{cfg['model_name']}-{cfg['date']}-{cfg['index']}"
    os.makedirs(f"model_weight/{cfg['model_name']}", exist_ok=True)
    os.makedirs(f"check_point/{cfg['model_name']}", exist_ok=True)

    wandb.init(
        project=cfg["wandb_project"],
        name=run_name,
        config=cfg,
    )

    # --- Training Loop ---
    best_val_loss = float("inf")
    early_stop_counter = 0
    saved_models = []
    task_weights = cfg["task_weights"]
    step = 0

    progress_bar = tqdm(range(num_total_steps), desc="Co-teaching Training", dynamic_ncols=True)

    for epoch in range(cfg["num_epochs"]):
        if early_stop_counter >= cfg["patience"]:
            print("Early stopping triggered. Training stopped.")
            break

        # Forget rate for this epoch
        forget_rate = get_forget_rate(epoch, cfg["noise_rate"], cfg["num_gradual"])
        print(f"\nEpoch {epoch+1}/{cfg['num_epochs']} | Forget Rate: {forget_rate:.4f} | Task Weights: {task_weights}")

        model1.train()
        model2.train()

        train_loss1_total = 0
        train_loss2_total = 0
        train_batch_count = 0

        for batch in train_loader:
            if early_stop_counter >= cfg["patience"]:
                print("Early stopping triggered during training.")
                break

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_tokens = batch["start_token"].to(device)
            end_tokens = batch["end_token"].to(device)

            targets = {
                "tag": batch["tag"].to(device),
                "time": batch["time"].to(device),
                "scale": batch["scale"].to(device),
                "negative": batch["negative"].to(device),
            }

            # Forward pass through both models
            outputs1 = model1(input_ids, attention_mask, start_tokens, end_tokens)
            outputs2 = model2(input_ids, attention_mask, start_tokens, end_tokens)

            # Co-teaching loss
            loss1, loss2, loss_details = compute_coteaching_loss(
                outputs1, outputs2, targets, forget_rate, task_weights,
            )

            # Update Model 1
            optimizer1.zero_grad()
            loss1.backward()
            optimizer1.step()
            scheduler1.step()

            # Update Model 2
            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()
            scheduler2.step()

            # Logging
            train_loss1_total += loss1.item()
            train_loss2_total += loss2.item()
            train_batch_count += 1

            step += 1
            epoch_progress = step / train_approx_batches

            progress_bar.set_postfix(
                loss1=f"{loss1.item():.4f}",
                loss2=f"{loss2.item():.4f}",
                epoch=f"{epoch_progress:.2f}",
                fr=f"{forget_rate:.3f}",
            )
            progress_bar.update(1)

            wandb.log({
                "step": step,
                "epoch": epoch_progress,
                "forget_rate": forget_rate,
                "loss1": loss1.item(),
                "loss2": loss2.item(),
                "lr_bert_m1": optimizer1.param_groups[0]['lr'],
                "lr_tag_head_m1": optimizer1.param_groups[1]['lr'],
                "lr_bert_m2": optimizer2.param_groups[0]['lr'],
            })

            # Periodic evaluation
            if step % cfg["eval_step"] == 0:
                save_coteaching_checkpoint(
                    model1, model2, optimizer1, optimizer2,
                    scheduler1, scheduler2, epoch, step,
                    f"check_point/{cfg['model_name']}/{cfg['date']}_{cfg['index']}_step{step}.pth",
                )

                avg_loss1 = train_loss1_total / train_batch_count if train_batch_count > 0 else 0
                avg_loss2 = train_loss2_total / train_batch_count if train_batch_count > 0 else 0

                # Reset train accumulators
                train_loss1_total = 0
                train_loss2_total = 0
                train_batch_count = 0

                # Validate with Model 1
                val_loss, val_losses = validate_coteaching(
                    model1, valid_loader, task_weights, device,
                    tag_loss_fn=tag_loss_fn, time_loss_fn=time_loss_fn,
                    scale_loss_fn=scale_loss_fn, neg_loss_fn=neg_loss_fn,
                )
                model1.train()
                model2.train()

                val_loss_dict = {f"val_loss_{k}": v for k, v in val_losses.items()}

                wandb.log({
                    "step": step,
                    "epoch_progress": epoch_progress,
                    "train_loss1": avg_loss1,
                    "train_loss2": avg_loss2,
                    "val_loss": val_loss,
                    **val_loss_dict,
                })

                print(f"\nStep {step} Epoch {epoch_progress:.2f}:")
                print(f"  Train Loss: M1={avg_loss1:.4f} M2={avg_loss2:.4f}")
                print(f"  Val Loss: {val_loss:.4f}")
                print(f"  Val Breakdown: {val_losses}")

                # Save best model
                if val_loss < best_val_loss:
                    print(f"  ✅ Val loss improved: {best_val_loss:.4f} → {val_loss:.4f}")
                    best_val_loss = val_loss
                    early_stop_counter = 0

                    # Save both models
                    model1_path = f"model_weight/{cfg['model_name']}/{cfg['date']}_{cfg['index']}_m1_step{step}.pt"
                    model2_path = f"model_weight/{cfg['model_name']}/{cfg['date']}_{cfg['index']}_m2_step{step}.pt"
                    torch.save(model1.state_dict(), model1_path)
                    torch.save(model2.state_dict(), model2_path)

                    saved_models.append((model1_path, model2_path))
                    if len(saved_models) > cfg["max_saved_models"]:
                        old_m1, old_m2 = saved_models.pop(0)
                        for p in [old_m1, old_m2]:
                            if os.path.exists(p):
                                os.remove(p)
                                print(f"  Removed old model: {p}")
                else:
                    early_stop_counter += 1
                    print(f"  ❌ No improvement. Early stop: {early_stop_counter}/{cfg['patience']}")

                if early_stop_counter >= cfg["patience"]:
                    print("Early stopping triggered. Training stopped.")
                    break

    wandb.finish()
    print("Training complete!")


if __name__ == "__main__":
    main()
