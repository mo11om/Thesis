"""
train_selfmix.py

Staff-level implementation of the SelfMix training loop for the Multi-Task SecBERT model.
Integrates Gaussian Mixture Model (GMM) dynamic separation, textual-level mixup, 
and KL-divergence consistency loss for robust learning against noisy labels.
"""
import os
import json
import torch
import wandb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import deque
from sklearn.mixture import GaussianMixture

# Import from your updated secbert_utils (Removed the notebook-specific variables)
from secbert_utils import (
    MultiTaskModel, ProcessedIterableDataset, compute_loss, validate_model, 
    save_checkpoint, load_checkpoint, compute_multitask_kl_loss,
    load_counter, CB_CE_Loss, FocalLoss, CLASSIFICATION_MISSING_VALUE
)

# ============================================================================
# Configuration & Hyperparameters
# ============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "secbert_selfmix"
DATE = "0426"
INDEX = 173
RUN_NAME = f"{MODEL_NAME}-{DATE}-{INDEX}"

NUM_EPOCHS = 30
BATCH_SIZE = 32
EVAL_STEP = 20000
PATIENCE = 5
MAX_SAVED_MODELS = 2

# SelfMix specific hyperparameters
ALPHA = 4.0               
TEMP = 0.5                
LAMBDA_P = 1.0            
LAMBDA_R = 1.0            
GMM_BUFFER_SIZE = 5000    
WARMUP_STEPS = 1000       

TASK_WEIGHTS = {"tag": 10.0, "time": 0.3, "scale": 0.2, "negative": 10.0}

LR_BERT = 1e-5
LR_TAG_HEAD = 5e-4
LR_TIME_HEAD = 1e-5
LR_SCALE_HEAD = 3e-5
LR_NEG_HEAD = 2e-5

# ============================================================================
# Dynamic Label Lists & Loss Setup (Reconstructed from secbert_smaller.ipynb)
# ============================================================================

# 1. Base classification lists
time_list = ['instant; past', 'instant; current', 'instant; future', 'period; past', 'period; current', 'period; future', 'period; past_current', 'period; current_future', 'period; past_future']
time2id = {time: idx for idx, time in enumerate(time_list)}

scale_list = [str(i) for i in range(-12, 13)]
scale2id = {scale: idx for idx, scale in enumerate(scale_list)}

# 2. Tag lists (Adjust the TAG_COUNT_PATH if you execute this from a different working directory)
TAG_COUNT_PATH = '../processed_data_task1_smaller/counter/tag_count_train_400k.json'
with open(TAG_COUNT_PATH, 'r', encoding='utf-8') as file:
    tag_counter = json.load(file)

tag_counter = dict(sorted(tag_counter.items(), key=lambda item: item[1], reverse=True))
count_threshold = 10
standard_rare_tags = {tag for tag, count in tag_counter.items() if count < count_threshold}
tag_list = [tag for tag in tag_counter.keys() if tag not in standard_rare_tags]
tag2id = {tag: idx for idx, tag in enumerate(tag_list)}

# 3. Task-specific Loss Functions Setup

# Tag Loss
train_tag_counts = load_counter("tag")
train_tag_counts.pop('-100', None)
num_tag_samples = [train_tag_counts.get(str(tag2id[tag]), 0) for tag in tag_list]
tag_loss_fn = CB_CE_Loss(num_tag_samples, beta=0.99)

# Time Loss
train_time_counts = load_counter("time")
train_time_counts.pop('-100', None)
total_samples = sum(train_time_counts.values())
num_classes = len(train_time_counts)

time_class_weights = {
    cls: total_samples / (num_classes * count) 
    for cls, count in train_time_counts.items()
}
time_class_weights = dict(sorted(time_class_weights.items(), key=lambda item: item[0]))
time_smoothed_weights = np.log1p(list(time_class_weights.values()))
time_smoothed_weights = np.clip(time_smoothed_weights, 1, None)
time_smoothed_weights[0] = 1.5
time_smoothed_weights[6] = 1.5
time_class_weights_tensor = torch.tensor(time_smoothed_weights, dtype=torch.float).to(DEVICE)
time_loss_fn = nn.CrossEntropyLoss(weight=time_class_weights_tensor, ignore_index=CLASSIFICATION_MISSING_VALUE)

# Scale Loss
scale_class_weights_dict = {15: 1.2, 21: 1.5}
weights_list = [scale_class_weights_dict.get(i, 1.0) for i in range(len(scale_list))]
scale_class_weights_tensor = torch.tensor(weights_list, dtype=torch.float).to(DEVICE)
scale_loss_fn = nn.CrossEntropyLoss(weight=scale_class_weights_tensor, ignore_index=CLASSIFICATION_MISSING_VALUE)

# Negative Loss
neg_loss_fn = FocalLoss(alpha=0.25, gamma=3.0, reduction="mean")
# ============================================================================
# Data Loading
# ============================================================================

def setup_dataloaders():
    print("Setting up dataloaders...")
    train_files = ["processed_iterable_dataset/train_400k.jsonl"]
    valid_files = ["processed_iterable_dataset/valid_50k.jsonl"] 
    
    train_dataset = ProcessedIterableDataset(train_files)
    valid_dataset = ProcessedIterableDataset(valid_files)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    # Approx batches based on 400k dataset / 64 batch size
    train_approx_batches = 728821 // BATCH_SIZE 
    return train_loader, valid_loader, train_approx_batches

# ============================================================================
# Training Loop
# ============================================================================

def train():
    train_loader, valid_loader, train_approx_batches = setup_dataloaders()
    
    # 1. Initialize Model
    model = MultiTaskModel(
        "nlpaueb/sec-bert-base",
        num_tags=len(tag_list), 
        num_times=len(time_list),
        num_scales=len(scale_list)
    ).to(DEVICE)

    # 2. Setup Optimizer & Scheduler
    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": LR_BERT, "weight_decay": 1e-2},  
        {"params": model.tag_head.parameters(), "lr": LR_TAG_HEAD,  "weight_decay": 1e-2},  
        {"params": model.time_head.parameters(), "lr": LR_TIME_HEAD,  "weight_decay": 1e-2},  
        {"params": model.scale_head.parameters(), "lr": LR_SCALE_HEAD,  "weight_decay": 1e-2},
        {"params": model.negative_head.parameters(), "lr": LR_NEG_HEAD,  "weight_decay": 1e-2},  
    ])
    
    num_total_steps = NUM_EPOCHS * train_approx_batches
    scheduler = CosineAnnealingLR(optimizer, T_max=num_total_steps // 4, eta_min=1e-7)

    # 3. Setup W&B and Checkpointing directories
    os.makedirs(f'model_weight/{MODEL_NAME}', exist_ok=True)
    os.makedirs(f'check_point/{MODEL_NAME}', exist_ok=True)

    wandb.init(
        project="multi-task-model",
        name=f'{RUN_NAME}_selfmix',
        config={
            "learning_rates": {"bert": LR_BERT, "tag_head": LR_TAG_HEAD, "time_head": LR_TIME_HEAD},
            "epochs": NUM_EPOCHS,
            "task_weights": TASK_WEIGHTS,
            "selfmix_alpha": ALPHA,
            "selfmix_temp": TEMP
        },
    )

    # 4. Training State
    best_val_loss = float("inf")
    early_stop_counter = 0 
    step = 0
    saved_models = []
    
    # SelfMix: Rolling buffer for dynamic GMM thresholding
    loss_buffer = deque(maxlen=GMM_BUFFER_SIZE)

    progress_bar = tqdm(range(num_total_steps), desc="Training", dynamic_ncols=True)

    for epoch in range(NUM_EPOCHS):
        if early_stop_counter >= PATIENCE:
            print("Early stopping triggered. Training stopped.")
            break

        model.train()
        train_loss = 0
        train_batch_count = 0
        
        for batch in train_loader:
            if early_stop_counter >= PATIENCE:
                break 
                
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            start_tokens = batch["start_token"].to(DEVICE)
            end_tokens = batch["end_token"].to(DEVICE)
            values = batch["value"].to(DEVICE)
            
            targets = {
                "tag": batch["tag"].to(DEVICE),
                "time": batch["time"].to(DEVICE),
                "scale": batch["scale"].to(DEVICE),
                "negative": batch["negative"].to(DEVICE)
            }
            
            # --- PHASE 1: Warmup & Rolling Evaluation ---
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids, attention_mask, start_tokens, end_tokens)
                # Primary task loss (tag) is used to gauge noise level
                batch_losses = F.cross_entropy(outputs["tag"], targets["tag"], ignore_index=-100, reduction='none')
                loss_buffer.extend(batch_losses.cpu().numpy().tolist())
                
            model.train()
            
            # --- PHASE 2: GMM Separation ---
            if step > WARMUP_STEPS and len(loss_buffer) == GMM_BUFFER_SIZE:
                # gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
                gmm = GaussianMixture(n_components=2, max_iter=50, tol=1e-3, reg_covar=1e-3)
                reshaped_losses = np.array(loss_buffer).reshape(-1, 1)
                gmm.fit(reshaped_losses)
                
                prob = gmm.predict_proba(batch_losses.cpu().numpy().reshape(-1, 1))
                clean_prob = prob[:, gmm.means_.argmin()] 
                clean_idx = torch.tensor(clean_prob > 0.5, dtype=torch.bool).to(DEVICE)
                noisy_idx = ~clean_idx
            else:
                # During warmup, assume all data is clean
                clean_idx = torch.ones(input_ids.size(0), dtype=torch.bool).to(DEVICE)
                noisy_idx = torch.zeros(input_ids.size(0), dtype=torch.bool).to(DEVICE)

            optimizer.zero_grad()
            
            # --- PHASE 3: Forward & Mixup Logic ---
            if noisy_idx.sum() == 0 or clean_idx.sum() == 0:
                # Standard routing if batch is homogenous
                embeddings = model.get_embeddings(input_ids, attention_mask, start_tokens, end_tokens)
                logits = model.classify(embeddings)
                loss, _ = compute_loss(logits, targets, values, TASK_WEIGHTS,
                                       tag_loss_fn=tag_loss_fn, time_loss_fn=time_loss_fn,
                                       scale_loss_fn=scale_loss_fn, neg_loss_fn=neg_loss_fn)
                loss.backward()
                optimizer.step()
                scheduler.step()
                final_loss = loss.item()
                
            else:
                # SelfMix Execution
                inputs_x_ids, inputs_x_mask = input_ids[clean_idx], attention_mask[clean_idx]
                start_x, end_x = start_tokens[clean_idx], end_tokens[clean_idx]
                targets_x = {k: v[clean_idx] for k, v in targets.items()}
                
                inputs_u_ids, inputs_u_mask = input_ids[noisy_idx], attention_mask[noisy_idx]
                start_u, end_u = start_tokens[noisy_idx], end_tokens[noisy_idx]
                
                # a) Generate pseudo-labels
                model.eval()
                with torch.no_grad():
                    out_u_raw = model(inputs_u_ids, inputs_u_mask, start_u, end_u)
                    targets_u = {}
                    for task in out_u_raw:
                        p = torch.softmax(out_u_raw[task], dim=1)
                        pt = p ** (1 / TEMP)  # Sharpening
                        targets_u[task] = (pt / pt.sum(dim=1, keepdim=True)).detach()
                
                # b) Extract embeddings (Train mode active for Dropout/Consistency)
                model.train()
                sents_x = model.get_embeddings(inputs_x_ids, inputs_x_mask, start_x, end_x)
                sents_u1 = model.get_embeddings(inputs_u_ids, inputs_u_mask, start_u, end_u)
                sents_u2 = model.get_embeddings(inputs_u_ids, inputs_u_mask, start_u, end_u) # 2nd pass for KL
                
                # c) Textual Level Mixup
                l = np.random.beta(ALPHA, ALPHA)
                l = max(l, 1 - l)
                min_size = min(sents_x.size(0), sents_u1.size(0))
                mixed_sents = l * sents_x[:min_size] + (1 - l) * sents_u1[:min_size]
                
                mixed_targets = {}
                for task in targets_x:
                    num_classes = out_u_raw[task].size(-1)
                    tx_valid_mask = targets_x[task][:min_size] != -100
                    safe_tx = targets_x[task][:min_size].clone()
                    safe_tx[~tx_valid_mask] = 0
                    tx_onehot = F.one_hot(safe_tx, num_classes=num_classes).float()
                    mixed_targets[task] = l * tx_onehot + (1 - l) * targets_u[task][:min_size]

                # d) Classification and Loss
                logits_mix = model.classify(mixed_sents)
                logits_u1 = model.classify(sents_u1)
                logits_u2 = model.classify(sents_u2)
                
                loss_mix, pse_loss = 0.0, 0.0
                for task in logits_mix:
                    w = TASK_WEIGHTS.get(task, 1.0)
                    loss_mix += w * -torch.mean(torch.sum(F.log_softmax(logits_mix[task], dim=-1) * mixed_targets[task], dim=-1))
                    pse_loss += w * (-torch.mean(F.log_softmax(logits_u1[task], dim=1).min(dim=1)[0]) * 0.5 
                                     -torch.mean(F.log_softmax(logits_u2[task], dim=1).min(dim=1)[0]) * 0.5)

                kl_loss = compute_multitask_kl_loss(logits_u1, logits_u2)
                total_loss = loss_mix + (pse_loss * LAMBDA_P) + (kl_loss * LAMBDA_R)
                
                total_loss.backward()
                optimizer.step()
                scheduler.step()
                
                final_loss = total_loss.item()
            
            # --- PHASE 4: Metrics & Logging ---
            train_loss += final_loss
            train_batch_count += 1
            step += 1
            epoch_progress = step / train_approx_batches
            
            progress_bar.set_postfix(loss=final_loss, epoch=f"{epoch_progress:.2f}")
            progress_bar.update(1)

            wandb.log({
                "step": step, 
                "epoch": epoch_progress,
                "loss": final_loss,
                "lr_bert": optimizer.param_groups[0]['lr'],
            })
            
            # --- PHASE 5: Evaluation & Checkpointing ---
            if step % EVAL_STEP == 0:
                save_checkpoint(model, optimizer, scheduler, epoch, step, f'check_point/{MODEL_NAME}/{DATE}_{INDEX}_step{step}.pth')
                
                avg_train_loss = train_loss / train_batch_count
                train_loss, train_batch_count = 0, 0
                
                val_loss, val_losses = validate_model(model, valid_loader, TASK_WEIGHTS, DEVICE, 
                                                      tag_loss_fn, time_loss_fn, scale_loss_fn, neg_loss_fn)
                model.train()
                
                wandb.log({
                    "step": step, 
                    "val_loss": val_loss,
                    **{f"val_loss_{k}": v for k, v in val_losses.items()}
                })
                
                print(f"\nStep {step} Epoch {epoch_progress:.2f}: Train Loss = {avg_train_loss:.4f} Validation Loss = {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    
                    model_path = f"model_weight/{MODEL_NAME}/{DATE}_{INDEX}_step{step}.pt"
                    torch.save(model.state_dict(), model_path)
                    saved_models.append(model_path)
                    
                    if len(saved_models) > MAX_SAVED_MODELS:
                        oldest_model = saved_models.pop(0) 
                        if os.path.exists(oldest_model):
                            os.remove(oldest_model)
                else:
                    early_stop_counter += 1
                    print(f"No improvement. Early stop counter: {early_stop_counter}/{PATIENCE}")

    wandb.finish()
    print("Training Complete.")

if __name__ == "__main__":
    # Ensure random seed is set for reproducibility (Assuming set_seed is in secbert_utils)
    # set_seed(42) 
    train()