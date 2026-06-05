import os
import json
import csv
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, precision_recall_fscore_support,
    mean_squared_error, mean_absolute_error, r2_score, mean_squared_log_error
)

# Custom Utils (Ensure secbert_utils is in your path)
from secbert_utils import (
    set_seed, convert_span_to_number, load_counter,
    process_batch, process_data, MultiTaskIterableDataset, 
    ProcessedIterableDataset, MultiTaskModel, hits_at_k, FocalLoss, CB_CE_Loss, 
    fact_loss_fn, compute_loss, save_checkpoint, load_checkpoint,
    validate_model, CLASSIFICATION_MISSING_VALUE, NUMERIC_MISSING_VALUE
)

print(f"PyTorch Version: {torch.__version__}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}")

# ==========================================
# 1. Configuration & Tokenizer Setup
# ==========================================
# REFACTORED: Swapped ModernBERT for BGE
BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5" # Use 'BAAI/bge-large-en-v1.5' if you need the larger variant
tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_NAME)

CLASSIFICATION_MISSING_VALUE = -100
NUMERIC_MISSING_VALUE = torch.finfo(torch.float32).max

# ==========================================
# 2. Vocabulary & Label Setup
# ==========================================
with open('../processed_data_task1_smaller/counter/tag_count_train_400k.json', 'r', encoding='utf-8') as file:
    tag_counter = json.load(file)
    
tag_counter = dict(sorted(tag_counter.items(), key=lambda item: item[1], reverse=True))
count_threshold = 10
standard_rare_tags = {tag for tag, count in tag_counter.items() if count < count_threshold}
tag_list = [tag for tag in tag_counter.keys() if tag not in standard_rare_tags]

id2tag = {idx: tag for idx, tag in enumerate(tag_list)}
tag2id = {tag: idx for idx, tag in enumerate(tag_list)}

time_list = ['instant; past', 'instant; current', 'instant; future', 'period; past', 'period; current', 'period; future', 'period; past_current', 'period; current_future', 'period; past_future']
id2time = {idx: time for idx, time in enumerate(time_list)}
time2id = {time: idx for idx, time in enumerate(time_list)}

scale_list = [str(i) for i in range(-12, 13)]
id2scale = {idx: scale for idx, scale in enumerate(scale_list)}
scale2id = {scale: idx for idx, scale in enumerate(scale_list)}

# ==========================================
# 3. Dataset & DataLoader
# ==========================================
train_files = ["processed_iterable_dataset_modern/train_400k.jsonl"]
valid_files = ["processed_iterable_dataset_modern/valid_50k.jsonl"]
# REFACTORED: Target data path provided
test_files = ["processed_iterable_dataset_bge/test_50k_filtered.jsonl"]
print(f"Test files: {test_files}")
train_dataset = ProcessedIterableDataset(train_files)
valid_dataset = ProcessedIterableDataset(valid_files)
test_dataset = ProcessedIterableDataset(test_files)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, drop_last=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, drop_last=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, drop_last=False)

def count_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)

train_total_samples = sum(count_lines(f) for f in train_files)
train_approx_batches = train_total_samples // batch_size
print(f"Train Expected Samples: {train_total_samples}, Batches: {train_approx_batches}")

# ==========================================
# 4. Counters & Loss Functions
# ==========================================
def get_local_counter(target_attr):
    target_path = f'processed_iterable_dataset/counter/secbert_train_small_{target_attr}.json'
    with open(target_path, "r", encoding='utf-8') as f:
        return json.load(f)

train_tag_counts = get_local_counter("tag")
train_tag_counts.pop('-100', None)
num_tag_samples = [train_tag_counts.get(str(tag2id[tag]), 0) for tag in tag_list]

train_time_counts = get_local_counter("time")
train_time_counts.pop('-100', None)
num_time_samples = [train_time_counts.get(str(time2id[time]), 0) for time in time_list]

train_neg_counts = get_local_counter("negative")
train_neg_counts.pop('-100', None)

train_scale_counts = get_local_counter("scale")
train_scale_counts.pop('-100', None)

# Time Loss
total_samples = sum(train_time_counts.values())
num_classes = len(train_time_counts)
time_class_weights = {cls: total_samples / (num_classes * count) for cls, count in train_time_counts.items()}
time_class_weights = list(dict(sorted(time_class_weights.items(), key=lambda item: item[0])).values())

time_smoothed_weights = np.log1p(time_class_weights)
time_smoothed_weights = np.clip(time_smoothed_weights, 1, None)
time_smoothed_weights[0] = 1.5
time_smoothed_weights[6] = 1.5
time_class_weights_tensor = torch.tensor(time_smoothed_weights, dtype=torch.float).to(device)

time_loss_fn = nn.CrossEntropyLoss(weight=time_class_weights_tensor, ignore_index=CLASSIFICATION_MISSING_VALUE)

# Scale Loss
scale_class_weights_dict = {15: 1.2, 21: 1.5}
weights_list = [scale_class_weights_dict.get(i, 1.0) for i in range(len(scale_list))]
scale_class_weights_tensor = torch.tensor(weights_list, dtype=torch.float).to(device)
scale_loss_fn = nn.CrossEntropyLoss(weight=scale_class_weights_tensor, ignore_index=CLASSIFICATION_MISSING_VALUE)

# Other Losses
tag_loss_fn = CB_CE_Loss(num_tag_samples, beta=0.99)
neg_loss_fn = FocalLoss(alpha=0.25, gamma=3.0, reduction="mean")

# ==========================================
# 5. Model Initialization
# ==========================================
# REFACTORED: Model tracking names and init
model_name = "bge"
date = "0426"
index = 173

model = MultiTaskModel(
    BGE_MODEL_NAME,
    num_tags=len(tag_list), 
    num_times=len(time_list),
    num_scales=len(scale_list)
)
model.to(device)

# ==========================================
# 6. Evaluation Logic
# ==========================================
def evaluate_model_pipeline(model, model_save_path, error_file_path, test_loader, task_weights, device, save_errors=False):
    os.environ["WANDB_DISABLED"] = "true"
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    
    all_losses = {'tag': 0, 'time': 0, 'fact': 0, 'scale': 0, 'negative': 0}
    total_hits = {"hits_1": 0, "hits_3": 0, "hits_5": 0}
    predictions = defaultdict(list)
    ground_truths = defaultdict(list)
    errors = defaultdict(list)
    test_batch_count = 0
    fact_res = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_tokens = batch["start_token"].to(device)
            end_tokens = batch["end_token"].to(device)
            values = batch["value"].to(device)
            
            targets = {
                "tag": batch["tag"].to(device),
                "time": batch["time"].to(device),
                "fact": batch["fact"].to(device),
                "scale": batch["scale"].to(device),
                "negative": batch["negative"].to(device)
            }
            
            outputs = model(input_ids, attention_mask, start_tokens, end_tokens)
            loss, losses, tag_hits_k = compute_loss(
                outputs, targets, values, task_weights, hits_k=True,
                tag_loss_fn=tag_loss_fn, time_loss_fn=time_loss_fn,
                scale_loss_fn=scale_loss_fn, neg_loss_fn=neg_loss_fn
            )
            test_batch_count += 1
            
            if tag_hits_k:
                for key in total_hits:
                    total_hits[key] += tag_hits_k[key]
                    
            for key in all_losses:
                if key in losses:
                    all_losses[key] += losses[key].item()
                
            for key in ["scale", "negative", "tag", "time"]:
                pred = torch.argmax(outputs[key], dim=-1).cpu().tolist()
                true = targets[key].cpu().tolist()

                for i, (p, t) in enumerate(zip(pred, true)):
                    predictions[key].append(p)
                    ground_truths[key].append(t)
                    if p != t:
                        errors[key].append({"batch_idx": batch_idx, "sample_idx": i, "true": t, "pred": p})

            # Fact & Numeric Processing
            classification_loss = nn.CrossEntropyLoss(ignore_index=CLASSIFICATION_MISSING_VALUE)
            if "fact" in targets:
                negative_pred = outputs["negative"].argmax(dim=-1).cpu().numpy()
                if "scale" in outputs:
                    scale_pred_class = outputs["scale"].argmax(dim=-1).cpu().numpy()
                else:
                    scale_pred_class = np.ones_like(values.cpu().numpy(), dtype=np.int64)
            
                id2scale_np = np.array([float(id2scale[idx]) for idx in range(len(id2scale))], dtype=np.float64)  
                scale_pred_values = id2scale_np[scale_pred_class]
                values_np = values.cpu().numpy()

                fact_pred = values_np * (-1) ** negative_pred * (10 ** scale_pred_values)
                fact_target = targets["fact"].view(-1).cpu().numpy()

                for i, (p, t, neg, scale, scale_val, val) in enumerate(zip(fact_pred, fact_target, negative_pred, scale_pred_class, scale_pred_values, values_np)):
                    if t == NUMERIC_MISSING_VALUE:
                        continue
                    predictions["fact"].append(p)
                    ground_truths["fact"].append(t)
                    fact_res.append({
                        "batch_idx": batch_idx, "sample_idx": i, "true": t, "pred": p,
                        "negative_pred": neg, "scale_pred": scale, "scale_val": scale_val, "value": val
                    })
                    if abs(p - t) > 1e-2:
                        errors["fact"].append({"batch_idx": batch_idx, "sample_idx": i, "true": t, "pred": p})

    for key in all_losses:
        all_losses[key] /= test_batch_count
    
    avg_hits_k = {key: total_hits[key] / test_batch_count for key in total_hits}
        
    if save_errors:
        os.makedirs(os.path.dirname(error_file_path), exist_ok=True)
        for key, error_list in errors.items():
            error_file = f"{error_file_path}_errors_{key}.csv"
            with open(error_file, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["batch_idx", "sample_idx", "true", "pred"])
                writer.writeheader()
                writer.writerows(error_list)
            print(f"🚨 Error samples saved to {error_file}")

    if fact_res:
        pd.DataFrame(fact_res).to_csv('fact_res.csv', index=False)
        
    return all_losses, dict(predictions), dict(ground_truths), avg_hits_k


# ==========================================
# 7. Execute Testing Pipeline
# ==========================================
task_weights = {"tag": 10.0, "time": 0.3, "scale": 0.2, "negative": 10.0}  

# REFACTORED: Checkpoint path provided
model_save_path = "model_weight/bge/0426_173_step140001.pt"
error_file_path = f'error_analysis/{model_name}_{date}_{index}'

print(f"Evaluating Model Checkpoint: {model_save_path}")
test_all_losses, test_predictions, test_ground_truths, avg_hits_k = evaluate_model_pipeline(
    model, model_save_path, error_file_path, test_loader, task_weights, device, save_errors=True
)

print("\nTest Losses:", test_all_losses)
print("Average Hits@K:", avg_hits_k)

os.makedirs("result", exist_ok=True)
for attr in ['scale', 'negative', 'tag', 'time', 'fact']:
    df = pd.DataFrame({
        f"true_{attr}": test_ground_truths[attr],
        f"pred_{attr}": test_predictions[attr],
    })
    df.to_csv(f"result/{model_name}_{date}_{index}_result_{attr}.csv", index=False, encoding="utf-8")
    print(f"Saved {attr} predictions to result folder.")

print("\nEvaluation Complete.")