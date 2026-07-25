# %% [markdown]
# ## Random Seed

# %%
import random
import torch
import numpy as np
import json
import os
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from transformers import AutoTokenizer, get_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from secbert_utils import (
    set_seed, convert_span_to_number, load_counter,
    process_batch, process_data, MultiTaskIterableDataset, 
    ProcessedIterableDataset,
    MultiTaskModel, hits_at_k, FocalLoss, CB_CE_Loss, 
    fact_loss_fn, compute_loss, save_checkpoint, load_checkpoint,
    validate_model, evaluate_model, CLASSIFICATION_MISSING_VALUE, 
    NUMERIC_MISSING_VALUE
)

# %%
print(f"PyTorch Version: {torch.__version__}")

# %% [markdown]
# ## Dataset

# %%
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-base")

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)

# 設定預設值
CLASSIFICATION_MISSING_VALUE = -100
NUMERIC_MISSING_VALUE = torch.finfo(torch.float32).max  # 3.4028235e+38

# %%
# IterableDataset

with open('../processed_data_task1_smaller/counter/tag_count_train_400k.json', 'r', encoding='utf-8') as file:
    tag_counter = json.load(file)
    
# 想讓數量多的類別在前面
tag_counter = dict(sorted(tag_counter.items(), key=lambda item:item[1], reverse=True))
print(f"Original tag count: {len(tag_counter)}")

count_threshold = 10
standard_rare_tags = {tag for tag, count in tag_counter.items() if count < count_threshold}
tag_list = [tag for tag in tag_counter.keys() if tag not in standard_rare_tags]

print(f"Top 5 tags: {tag_list[:5]}")
print(f'Length of standard_rare_tags: {len(standard_rare_tags)}')
print(f'Length of all tags: {len(tag_list)}')

id2tag = {idx: tag for idx, tag in enumerate(tag_list)}
tag2id = {tag: idx for idx, tag in enumerate(tag_list)}

time_list = ['instant; past', 'instant; current', 'instant; future', 'period; past', 'period; current', 'period; future', 'period; past_current', 'period; current_future', 'period; past_future']
id2time = {idx: time for idx, time in enumerate(time_list)}
time2id = {time: idx for idx, time in enumerate(time_list)}

scale_list = [str(i) for i in range(-12, 13)]
id2scale = {idx: scale for idx, scale in enumerate(scale_list)}
scale2id = {scale: idx for idx, scale in enumerate(scale_list)}

# %% [markdown]
# #### Load Processed Iterable Dataset

# %%
train_files = ["processed_iterable_dataset/train_400k.jsonl"]
valid_files = ["processed_iterable_dataset/valid_50k.jsonl"] 
test_files = ["processed_iterable_dataset/test_50k.jsonl"]

train_dataset = ProcessedIterableDataset(train_files)
valid_dataset = ProcessedIterableDataset(valid_files)
test_dataset = ProcessedIterableDataset(test_files)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, drop_last=False)

# %%
def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

# 計算訓練資料大約的 batch 數
train_total_samples = sum(count_lines(f) for f in train_files)
train_approx_batches = train_total_samples // batch_size

valid_total_samples = sum(count_lines(f) for f in valid_files)
valid_approx_batches = valid_total_samples // batch_size

test_total_samples = sum(count_lines(f) for f in test_files)
test_approx_batches = test_total_samples // batch_size

print(f"Train 預計數量: {train_total_samples}, 預計 {train_approx_batches} 個 batch")
print(f"Valid 預計數量: {valid_total_samples}, 預計 {valid_approx_batches} 個 batch")
print(f"Test 預計數量: {test_total_samples}, 預計 {test_approx_batches} 個 batch")

# %% [markdown]
# ## Count

# %%
# load counter result
def load_custom_counter(target_attr):
    target_path = f'processed_iterable_dataset/counter/secbert_train_small_{target_attr}.json'
    with open(target_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data

train_tag_counts = load_custom_counter("tag")
train_tag_counts.pop('-100', None)
num_tag_samples = [train_tag_counts.get(str(tag2id[tag]), 0) for tag in tag_list]
print(f'Total tag class: {len(train_tag_counts)}')

train_time_counts = load_custom_counter("time")
train_time_counts.pop('-100', None)
num_time_samples = [train_time_counts.get(str(time2id[time]), 0) for time in time_list]

train_neg_counts = load_custom_counter("negative")
train_neg_counts.pop('-100', None)
num_neg_samples = [train_neg_counts.get(neg, 0) for neg in sorted(train_neg_counts.keys())]

train_scale_counts = load_custom_counter("scale")
train_scale_counts.pop('-100', None)
num_scale_samples = [train_scale_counts.get(str(scale2id[scale]), 0) for scale in scale_list]

# %% [markdown]
# ## Loss Setup

# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# time loss
time_class_counts = torch.tensor(num_time_samples)
total_samples = sum(train_time_counts.values())
num_classes = len(train_time_counts)

time_class_weights = {
    cls: total_samples / (num_classes * count) 
    for cls, count in train_time_counts.items()
}
time_class_weights = dict(sorted(time_class_weights.items(), key=lambda item:item[0]))
time_class_weights = list(time_class_weights.values())

time_smoothed_weights = np.log1p(time_class_weights)
MIN_WEIGHT = 1 # 設定最小值
time_smoothed_weights = np.clip(time_smoothed_weights, MIN_WEIGHT, None)
time_smoothed_weights[0] = 1.5
time_smoothed_weights[6] = 1.5

time_class_weights_tensor = torch.tensor(time_smoothed_weights, dtype=torch.float).to(device)
time_loss_fn_ce = nn.CrossEntropyLoss(weight=time_class_weights_tensor, ignore_index=CLASSIFICATION_MISSING_VALUE)

# scale loss
scale_class_weights = {15: 1.2,  21: 1.5}
num_classes = len(scale_list)
weights_list = [scale_class_weights.get(i, 1.0) for i in range(num_classes)]

scale_class_weights_tensor = torch.tensor(weights_list, dtype=torch.float).to(device)
scale_loss_fn = nn.CrossEntropyLoss(weight=scale_class_weights_tensor, ignore_index=CLASSIFICATION_MISSING_VALUE)

# negative loss
neg_loss_fn = FocalLoss(alpha=0.25, gamma=3.0, reduction="mean")

# tag and time loss
tag_loss_fn = CB_CE_Loss(num_tag_samples, beta=0.99)
time_loss_fn = CB_CE_Loss(num_time_samples)

# %% [markdown]
# ## Train

# %% [markdown]
# #### Init & Setting

# %%
def get_task_weights(epoch):
    if epoch < 3:
       return {"tag": 15.0, "time": 1.0, "scale": 0.1, "negative": 3}  
    elif epoch < 8:
        return {"tag": 10.0, "time": 1.2, "scale": 0.2, "negative": 5}          
    elif epoch < 12:
        return {"tag": 10.0, "time": 1.5, "scale": 0.2, "negative": 5}  
    else:
        return {"tag": 8.0, "time": 1.0, "scale": 0.2, "negative": 3}

# %%
num_warmup_steps = 5
num_epochs = 30
eval_step = 20000 
task_weights = {"tag": 10.0, "time": 0.3, "scale": 0.2, "negative": 10.0}  
patience = 5
max_saved_models = 2
saved_models = [] 

model_name = "secbert"
date = "0426"
index = 173
run_name = f"{model_name}-{date}-{index}"

model = MultiTaskModel(
    "nlpaueb/sec-bert-base",
    num_tags=len(tag_list), 
    num_times=len(time_list),
    num_scales=len(scale_list)
)

bert_lr = 1e-5
tag_head_lr = 5e-4
time_head_lr = 1e-5
scale_head_lr = 3e-5
negative_head_lr = 2e-5

optimizer = AdamW([
    {"params": model.bert.parameters(), "lr": bert_lr, "weight_decay": 1e-2},  
    {"params": model.tag_head.parameters(), "lr": tag_head_lr,  "weight_decay": 1e-2},  
    {"params": model.time_head.parameters(), "lr": time_head_lr,  "weight_decay": 1e-2},  
    {"params": model.scale_head.parameters(), "lr": scale_head_lr,  "weight_decay": 1e-2},
    {"params": model.negative_head.parameters(), "lr": negative_head_lr,  "weight_decay": 1e-2},  
])

num_total_steps = num_epochs * train_approx_batches
scheduler = CosineAnnealingLR(optimizer, T_max=num_total_steps // 4, eta_min=1e-7)

best_val_loss = float("inf")
early_stop_counter = 0 

os.makedirs(f'model_weight/{model_name}', exist_ok=True)
os.makedirs(f'check_point/{model_name}', exist_ok=True)

wandb.init(
    project="multi-task-model",
    resume="allow",
    id='wqugd5cx',
    name=f'{run_name}_cont_0330_173',
    config={
        "learning_rates": { 
            "bert": bert_lr,
            "tag_head": tag_head_lr,
            "time_head": time_head_lr,
            "scale_head": scale_head_lr,
            "negative_head": negative_head_lr
        },
        "epochs": num_epochs,
        "task_weights": task_weights,
        "train_data_size": train_total_samples,
        "valid_data_size": valid_total_samples,
        "eval_step": eval_step
    },
)

# %% [markdown]
# #### Validation

# %%
def validate_model_eval(model, val_loader, task_weights, device):
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
                "negative": batch["negative"].to(device)
            }
            
            outputs = model(input_ids, attention_mask, start_tokens, end_tokens)
            loss, losses = compute_loss(outputs, targets, values, task_weights, 
                                        tag_loss_fn=tag_loss_fn,
                                        time_loss_fn=time_loss_fn,
                                        scale_loss_fn=scale_loss_fn,
                                        neg_loss_fn=neg_loss_fn)
            
            val_loss += loss.item()
            valid_batch_count += 1
            
            for key in all_losses:
                all_losses[key] += losses.get(key, 0)

    val_loss /= valid_batch_count
    for key in all_losses:
        all_losses[key] /= valid_batch_count
    
    return val_loss, all_losses

# %% [markdown]
# #### Training

# %%
model = model.to(device)
progress_bar = tqdm(range(num_total_steps), desc="Training", dynamic_ncols=True)
global_step = 0

for epoch in range(num_epochs):
    if early_stop_counter >= patience:
        print(f"Early stopping triggered (Counter: {early_stop_counter}). Training stopped.")
        break
        
    model.train()
    print(f"\n--- Starting Epoch {epoch + 1}/{num_epochs} ---")
    print(f"Task weights applied: {task_weights}")
    
    train_loss = 0
    train_batch_count = 0
    train_losses = {}
    
    # We use enumerate starting at 1 to cleanly track the epoch step
    for epoch_step, batch in enumerate(train_loader, 1):
        if early_stop_counter >= patience:
            print("Early stopping triggered during training. Stopping current epoch.")
            break 
            
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_tokens = batch["start_token"].to(device)
        end_tokens = batch["end_token"].to(device)
        values = batch["value"].to(device)
        
        targets = {
            "tag": batch["tag"].to(device),
            "time": batch["time"].to(device),
            "scale": batch["scale"].to(device),
            "negative": batch["negative"].to(device)
        }

        outputs = model(input_ids, attention_mask, start_tokens, end_tokens)
        loss, losses = compute_loss(outputs, targets, values, task_weights,
                                    tag_loss_fn=tag_loss_fn,
                                    time_loss_fn=time_loss_fn,
                                    scale_loss_fn=scale_loss_fn,
                                    neg_loss_fn=neg_loss_fn)     
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Accumulate metrics
        train_loss += loss.item()
        for key, val in losses.items():
            train_losses[key] = train_losses.get(key, 0) + val
            
        train_batch_count += 1
        global_step += 1
        epoch_progress = global_step / train_approx_batches
        
        # Update progress bar with epoch and epoch_step info
        progress_bar.set_postfix(
            epoch=f"{epoch + 1}/{num_epochs}",
            epoch_step=f"{epoch_step}/{train_approx_batches}",
            loss=f"{loss.item():.4f}"
        )
        progress_bar.update(1)

        # Log metrics to WandB
        wandb.log({
            "global_step": global_step,
            "epoch_progress": epoch_progress,
            "epoch": epoch + 1,
            "epoch_step": epoch_step,
            "lr_bert": optimizer.param_groups[0]['lr'],
            "lr_tag_head": optimizer.param_groups[1]['lr'],
            "lr_time_head": optimizer.param_groups[2]['lr'],
            "lr_scale_head": optimizer.param_groups[3]['lr'],
            "lr_negative_head": optimizer.param_groups[4]['lr'],
            "batch_loss": loss.item()
        })        
    
        # Evaluation Step
        if global_step % eval_step == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, f'check_point/{model_name}/{date}_{index}_step{global_step}.pth')
            
            avg_train_loss = train_loss / train_batch_count if train_batch_count > 0 else 0
            train_loss_dict = {f"train_loss_{key}": train_losses[key] / train_batch_count for key in train_losses} if train_batch_count > 0 else {}
            
            # Reset train loss accumulators after evaluation
            train_loss = 0
            train_batch_count = 0
            train_losses = {}

            val_loss, val_losses = validate_model_eval(model, valid_loader, task_weights, device)
            val_loss_dict = {f"val_loss_{key}": val_losses[key] for key in val_losses}
            model.train()
            
            wandb.log({
                "global_step": global_step, 
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                **train_loss_dict,
                **val_loss_dict
            })
    
            print(f"\n[{model_name}] Epoch {epoch + 1}/{num_epochs} | Epoch Step {epoch_step}/{train_approx_batches} | Global Step {global_step}")
            print(f"Train Loss = {avg_train_loss:.4f} | Validation Loss = {val_loss:.4f}")
            print(f"Validation Loss Breakdown: {val_losses}")
                
            # Save model if validation loss improves
            if val_loss < best_val_loss:
                print(f"--> Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                best_val_loss = val_loss
                early_stop_counter = 0
                
                model_save_path_epoch = f"model_weight/{model_name}/{date}_{index}_step{global_step}.pt"
                torch.save(model.state_dict(), model_save_path_epoch)
                saved_models.append(model_save_path_epoch)
                
                # Keep only max_saved_models
                if len(saved_models) > max_saved_models:
                    oldest_model = saved_models.pop(0) 
                    if os.path.exists(oldest_model):
                        os.remove(oldest_model)
                        print(f"Removed old model: {oldest_model}")
            else:
                early_stop_counter += 1
                print(f"--> No improvement. Early stop counter: {early_stop_counter}/{patience}")
        
            # Early stopping check inside inner loop
            if early_stop_counter >= patience:
                print("Early stopping triggered during evaluation. Stopping training.")
                break

wandb.finish()