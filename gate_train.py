# %% [markdown]
# # setup

# %% [markdown]
# ## Random Seed

# %%
import random
import torch
import numpy as np
import json
from shared_utils import set_seed

with open('config.json', 'r') as f:
    config = json.load(f)

# %%
print(torch.__version__)

# %%

# Set random seed
set_seed(config['seed'])


# %% [markdown]
# ## Dataset

# %%
from transformers import AutoTokenizer
import torch

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-pretrain", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained(config['model']['base_model'])

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)

# 設定預設值
CLASSIFICATION_MISSING_VALUE = -100
NUMERIC_MISSING_VALUE = torch.finfo(torch.float32).max  # 3.4028235e+38

# %%
# batch processing - imports and usage from shared_utils
from shared_utils import process_batch, process_batch_wrapper, process_data


# %%
# IterableDataset

with open('../processed_data_task1_smaller/counter/tag_count_train_400k.json', 'r', encoding = 'utf-8') as file:
# with open('processed_iterable_dataset/counter/train_8k.json', 'r', encoding = 'utf-8') as file:
# with open('../processed_data_task1/counter/tag_count_train_data.json', 'r', encoding = 'utf-8') as file:    
    tag_counter = json.load(file)
    
# 想讓數量多的類別在前面

tag_counter = dict(sorted(tag_counter.items(), key = lambda item:item[1], reverse=True))
print(len(tag_counter))
count_threshold = 10
standard_rare_tags = {tag for tag, count in tag_counter.items() if count < count_threshold}
tag_list = [tag for tag in tag_counter.keys() if tag not in standard_rare_tags]
print(tag_list[:5])
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
# with open('processed_data_task1_smaller/counter/train_100k.json')

# %%
from shared_utils import convert_span_to_number


# %% [markdown]
# #### IterableDataset

# %%
# train_files = ["../processed_data_task1_smaller/train_400k.jsonl"]
# valid_files = ["../processed_data_task1_smaller/valid_50k.jsonl"]
# test_files = ["../processed_data_task1_smaller/test_50k.jsonl"]

# %%
import random
from torch.utils.data import DataLoader
from shared_utils import MultiTaskIterableDataset, BufferedShuffleDataset,ProcessedIterableDataset


# %%


# %%
batch_size = config['data']['batch_size']
# num_workers = 4
# # target_attrs = ["tag", "time", "scale", "negative", "fact"]

# # train_dataset = MultiTaskIterableDataset(
# #     files = ["../processed_data_task1_smaller/train_400k.jsonl"], 
# #     # files = ["processed_data_task1/train_data_shuffled.jsonl"],
# #     standard_rare_tags= standard_rare_tags,
# #     target_attrs = target_attrs,
# #     tokenizer = tokenizer,
# #     tag2id=tag2id,
# #     time2id=time2id,
# #     scale2id=scale2id,
# #     CLASSIFICATION_MISSING_VALUE=CLASSIFICATION_MISSING_VALUE,
# #     NUMERIC_MISSING_VALUE=NUMERIC_MISSING_VALUE
# #     )

# # valid_dataset = MultiTaskIterableDataset(
# #     files = ["../processed_data_task1_smaller/valid_50k.jsonl"], 
# #     # files = ["processed_data_task1/valid_data.jsonl"], 
# #     standard_rare_tags= standard_rare_tags,
# #     target_attrs = target_attrs,
# #     tokenizer = tokenizer,
# #     tag2id=tag2id,
# #     time2id=time2id,
# #     scale2id=scale2id,
# #     CLASSIFICATION_MISSING_VALUE=CLASSIFICATION_MISSING_VALUE,
# #     NUMERIC_MISSING_VALUE=NUMERIC_MISSING_VALUE
# #     )

# # test_dataset = MultiTaskIterableDataset(
# #     files = ["../processed_data_task1_smaller/test_50k.jsonl"], 
# #     # files = ["processed_data_task1/test_data.jsonl"], 
# #     standard_rare_tags= standard_rare_tags,
# #     target_attrs = target_attrs,
# #     tokenizer = tokenizer,
# #     tag2id=tag2id,
# #     time2id=time2id,
# #     scale2id=scale2id,
# #     CLASSIFICATION_MISSING_VALUE=CLASSIFICATION_MISSING_VALUE,
# #     NUMERIC_MISSING_VALUE=NUMERIC_MISSING_VALUE
# #     )

# train_dataset = ProcessedIterableDataset(train_files)

# valid_dataset = ProcessedIterableDataset(valid_files)

# test_dataset = ProcessedIterableDataset(test_files)

# train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers)
# valid_dataloader = DataLoader(valid_dataset, batch_size = batch_size, num_workers = num_workers)
# test_dataloader = DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers)

# %% [markdown]
# #### Load Processed Iterable Dataset

# %%

train_files = config['data']['train_files']
valid_files = config['data']['valid_files'] 
test_files = config['data']['test_files']

# %%
# Read pre-processed JSONL files
from torch.utils.data import DataLoader
from shared_utils import ProcessedIterableDataset


# %%
train_dataset = ProcessedIterableDataset(train_files)

valid_dataset = ProcessedIterableDataset(valid_files)

test_dataset = ProcessedIterableDataset(test_files)

batch_size = config['data']['batch_size']
num_workers = config['data']['num_workers']
# train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 4, drop_last = False)
# valid_loader = DataLoader(valid_dataset, batch_size = batch_size, num_workers = 4, drop_last = False)
# test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers = 4, drop_last = False)

train_loader = DataLoader(train_dataset, batch_size = batch_size, num_workers = num_workers)
valid_loader = DataLoader(valid_dataset, batch_size = batch_size, num_workers = num_workers)
test_loader = DataLoader(test_dataset, batch_size = batch_size, num_workers = num_workers)

# %%
import os
from shared_utils import count_lines

# Calculate approximate batch counts
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

# %% [markdown]
# #### 計算 tag, scale, negative, time 的類別個數
# 計算後存成 JSON，之後可以直接用

# %%
# load counter result
from shared_utils import load_counter

train_tag_counts = load_counter("tag")
train_tag_counts.pop('-100', None)
num_tag_samples = [train_tag_counts.get(str(tag2id[tag]), 0) for tag in tag_list]
print(f'Total tag class: {len(train_tag_counts)}')
train_time_counts = load_counter("time")
train_time_counts.pop('-100', None)
num_time_samples = [train_time_counts.get(str(time2id[time]), 0) for time in time_list]
print(train_time_counts)

train_neg_counts = load_counter("negative")
train_neg_counts.pop('-100', None)
num_neg_samples = [train_neg_counts.get(neg, 0) for neg in sorted(train_neg_counts.keys())]
print(train_neg_counts)

train_scale_counts = load_counter("scale")
train_scale_counts.pop('-100', None)
num_scale_samples = [train_scale_counts.get(str(scale2id[scale]), 0) for scale in scale_list]
print(train_scale_counts)


# %% [markdown]
# ## Model

# %%


# %%
import torch
import torch.nn as nn
from shared_utils import GateHead, MultiTaskModel


# %%
# MultiTaskModel is imported from shared_utils
# See GateHead and MultiTaskModel in shared_utils module


# %% [markdown]
# ## Loss

# %%
# Hits@K metric
import torch
from shared_utils import hits_at_k


# %%
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# device = 'cpu'

# %%
# time loss
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np

# 取得 time 類別的出現次數
time_class_counts = torch.tensor(num_time_samples)

total_samples = sum(train_time_counts.values())
num_classes = len(train_time_counts)

time_class_weights = {
    cls: total_samples / (num_classes * count) 
    for cls, count in train_time_counts.items()
}
time_class_weights = dict(sorted(time_class_weights.items(), key = lambda item:item[0]))
time_class_weights = list(time_class_weights.values())

time_smoothed_weights = np.log1p(time_class_weights)

MIN_WEIGHT =  config['loss']['time']['min_weight'] # 設定最小值
time_smoothed_weights = np.clip(time_smoothed_weights, MIN_WEIGHT, None)
for idx, weight in config['loss']['time']['special_weights'].items():
    time_smoothed_weights[int(idx)] = weight
print(time_smoothed_weights)
time_class_weights = torch.tensor(time_smoothed_weights, dtype=torch.float).to(device)
time_loss_fn = nn.CrossEntropyLoss(weight = time_class_weights, ignore_index = CLASSIFICATION_MISSING_VALUE)

# %%
# scale loss
scale_class_weights = {int(k): v for k, v in config['loss']['scale']['weights'].items()}
num_classes = len(scale_list)
weights_list = [scale_class_weights.get(i, 1.0) for i in range(num_classes)]

scale_class_weights = torch.tensor(weights_list, dtype=torch.float).to(device)

# 定義 loss function
scale_loss_fn = nn.CrossEntropyLoss(weight=scale_class_weights, ignore_index=CLASSIFICATION_MISSING_VALUE)

# %%
# # netagive loss

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     '''https://doi.org/10.1109/tpami.2018.2858826'''
#     def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", ignore_index = CLASSIFICATION_MISSING_VALUE):
#         """
#         alpha: 平衡因子 (適用於正負類不平衡)
#         gamma: 縮放因子 (讓難分類的樣本 loss 權重變高)
#         reduction: 可選 ["mean", "sum", "none"]，控制 loss 的輸出方式
#         """
#         super(FocalLoss, self).__init__()
#         if isinstance(alpha, (float, int)):  # 確保 alpha 是 tensor
#             self.alpha = torch.tensor([1 - alpha, alpha])  # alpha_neg, alpha_pos
#         else:
#             self.alpha = torch.tensor(alpha) 
#         self.gamma = gamma
#         self.reduction = reduction
#         self.ce_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=CLASSIFICATION_MISSING_VALUE)
#         self.ignore_index = ignore_index

#     def forward(self, logits, targets):
#         """
#         logits: 預測值 (模型輸出，形狀 [batch_size, 2]，未經 softmax)
#         targets: 標籤值 (形狀 [batch_size]，0 或 1)
#         """
#         device = logits.device
#         self.alpha = self.alpha.to(device)
#         # print('focal loss', targets)
#         # print(targets.shape)
#         if targets.dim() > 1:
#             targets = targets.argmax(dim=-1)
        
#          # 1. 移除 ignore_index
#         valid_mask = (targets != self.ignore_index)
#         targets = targets[valid_mask]
#         logits = logits[valid_mask]

#         # if targets.numel() == 0:  # 避免 loss 計算時出現空值
#         #     return torch.tensor(0.0, device=device, requires_grad=True)
#         # if torch.any(filtered_targets < 0) or torch.any(filtered_targets >= logits.shape[-1]):
#         #     raise ValueError(f"Invalid target values detected: {filtered_targets}")
        
#         # 2. 計算 CrossEntropyLoss
#         ce_loss = self.ce_loss(logits, targets)  # 計算 cross entropy loss
#         pt = torch.exp(-ce_loss)  # 選擇正確類別的機率
        
#         # 4. 計算 focal loss 權重
#         focal_weight = (1 - pt) ** self.gamma  # (1 - p_t)^gamma
#         alpha_weight = self.alpha.gather(0, targets.data.view(-1))  # 根據 targets 索引 alpha

#         loss = alpha_weight * focal_weight * ce_loss

#         # 5. 根據 reduction 返回 loss
#         if self.reduction == "mean":
#             return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=device, requires_grad=True)
#         elif self.reduction == "sum":
#             return loss.sum()
#         else:
#             return loss  # 不做平均，返回 batch loss

# neg_loss = FocalLoss(alpha=0.25, gamma=3.0, reduction="mean")

# %%
# # tag loss
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class CB_CE_Loss(nn.Module):
#     '''
#     https://ieeexplore.ieee.org/abstract/document/8953804
#     '''
#     def __init__(self, num_samples, beta=0.99, ignore_index=CLASSIFICATION_MISSING_VALUE):
#         """
#         Args:
#             num_samples: list or tensor, 每個類別的樣本數
#             beta: 控制 class-balanced 權重的超參數 (通常取 0.99)
#         """
#         super(CB_CE_Loss, self).__init__()
        
#         # 計算 Class-Balanced 權重
#         effective_num = 1.0 - torch.pow(torch.tensor(beta), torch.tensor(num_samples))
#         weights = (1.0 - beta) / (effective_num + 1e-8)
#         # self.weights = weights / torch.sum(weights)  # normalize
#         self.weights = weights
#         self.ignore_index = ignore_index
        
#     def forward(self, logits, targets):
#         """
#         Args:
#             logits: (batch_size, num_classes) 模型輸出的 logits
#             targets: (batch_size,) 類別索引標籤
#         Returns:
#             CB-CE Loss 值
#         """
        
#         if targets.dim() > 1:
#             targets = targets.argmax(dim=-1)
            
#         valid_mask = (targets != self.ignore_index)  # 只對有效的 targets 計算 loss
#         targets = targets[valid_mask]
#         logits = logits[valid_mask]
        
#         # 計算標準 CE Loss
#         ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        
#         # 依照類別權重調整 loss
#         class_weights = self.weights.to(logits.device)
#         weighted_loss = ce_loss * class_weights[targets]
        
#         # weighted_loss = ce_loss * class_weights[targets] * weight_mask.float()
#         return torch.mean(weighted_loss)

#         # return weighted_loss.sum() / weight_mask.sum()  # 只對有效樣本取平均

# # train_tag_counts = get_value_counts(train_loader, "tag")
# # num_tag_samples = [train_tag_counts.get(tag, 0) for tag in sorted(train_tag_counts.keys())]  # 確保對應到索引順序
# tag_loss_fn = CB_CE_Loss(num_tag_samples, beta = 0.99)

# %%
# # train_time_counts = get_value_counts(train_loader, "time")
# # num_time_samples = [train_time_counts.get(time, 0) for time in sorted(train_time_counts.keys())]
# time_loss_fn = CB_CE_Loss(num_time_samples)

# %%
# Loss functions imported from shared_utils
from shared_utils import huber_loss, signed_log


# %%
import torch
import torch.nn as nn
from shared_utils import FocalLoss, CB_CE_Loss


# %%
# --- Tag Loss ---
# (Assumes you have num_tag_samples defined as before)
tag_loss_fn = CB_CE_Loss(num_tag_samples, beta=config['loss']['tag']['beta'], ignore_index=CLASSIFICATION_MISSING_VALUE)

# --- Time Loss ---
# (Assumes you have time_class_weights defined as before)
# Note: standard CrossEntropyLoss supports reduction='none' by default
time_loss_fn = nn.CrossEntropyLoss(weight=time_class_weights, ignore_index=CLASSIFICATION_MISSING_VALUE, reduction='none')

# --- Scale Loss ---
# (Assumes scale_class_weights defined)
scale_loss_fn = nn.CrossEntropyLoss(weight=scale_class_weights, ignore_index=CLASSIFICATION_MISSING_VALUE, reduction='none')

# --- Negative Loss ---
neg_loss = FocalLoss(alpha=config['loss']['negative']['alpha'], gamma=config['loss']['negative']['gamma'], reduction="none")

# %%
# # loss function

# classification_loss = nn.CrossEntropyLoss(ignore_index = CLASSIFICATION_MISSING_VALUE)
# mse_loss = nn.MSELoss()

# def compute_loss(outputs, targets, values, task_weights = None, hits_k = False):
#     losses = {}
#     tag_hits_k = {}
#     if "tag" in targets:
#         losses["tag"] = tag_loss_fn(outputs["tag"], targets["tag"])

#         # if not model.training:
#         if hits_k:
#             tag_hits_k["hits_1"] = hits_at_k(outputs["tag"], targets["tag"], k = 1)
#             tag_hits_k["hits_3"] = hits_at_k(outputs["tag"], targets["tag"], k = 3)
#             tag_hits_k["hits_5"] = hits_at_k(outputs["tag"], targets["tag"], k = 5)

#     if "time" in targets:
#         losses["time"] = time_loss_fn(outputs["time"], targets["time"])
    
#     if "scale" in targets:
#         losses["scale"] = scale_loss_fn(outputs["scale"], targets["scale"])
    
#     if "negative" in targets:
#         # 負號預測
#         negative_pred = outputs["negative"].argmax(dim=-1)  # [batch_size]
#         losses["negative"] = neg_loss(outputs["negative"], targets["negative"])
    
#     total_loss = sum(task_weights[k] * losses[k] for k in losses.keys())
    
#     # 平衡 loss 避免變大
#     total_loss = total_loss / sum(task_weights.values())

#     # print('total_loss', total_loss)
#     return (total_loss, losses, tag_hits_k) if hits_k else (total_loss, losses)

# %%
# --- Tag Loss ---
# (Assumes you have num_tag_samples defined as before)
tag_loss_fn = CB_CE_Loss(num_tag_samples, beta=config['loss']['tag']['beta'], ignore_index=CLASSIFICATION_MISSING_VALUE)

# --- Time Loss ---
# (Assumes you have time_class_weights defined as before)
# Note: standard CrossEntropyLoss supports reduction='none' by default
time_loss_fn = nn.CrossEntropyLoss(weight=time_class_weights, ignore_index=CLASSIFICATION_MISSING_VALUE, reduction='none')

# --- Scale Loss ---
# (Assumes scale_class_weights defined)
scale_loss_fn = nn.CrossEntropyLoss(weight=scale_class_weights, ignore_index=CLASSIFICATION_MISSING_VALUE, reduction='none')

# --- Negative Loss ---
neg_loss_fn = FocalLoss(alpha=config['loss']['negative']['alpha'], gamma=config['loss']['negative']['gamma'], reduction="none")

# %%
CLASSIFICATION_MISSING_VALUE=-100
classification_loss = nn.CrossEntropyLoss(ignore_index = CLASSIFICATION_MISSING_VALUE)
mse_loss = nn.MSELoss()


# %%
from shared_utils import compute_loss


# %%
# def compute_loss(outputs, targets, values, task_weights = None, hits_k = False):
#     losses = {}
#     tag_hits_k = {}
#     if "tag" in targets:
#         losses["tag"] = tag_loss_fn(outputs["tag"], targets["tag"])

#         if hits_k:
#             tag_hits_k["hits_1"] = hits_at_k(outputs["tag"], targets["tag"], k = 1)
#             tag_hits_k["hits_3"] = hits_at_k(outputs["tag"], targets["tag"], k = 3)
#             tag_hits_k["hits_5"] = hits_at_k(outputs["tag"], targets["tag"], k = 5)

#     if "time" in targets:
#         losses["time"] = time_loss_fn(outputs["time"], targets["time"])
    
#     if "scale" in targets:
#         losses["scale"] = scale_loss_fn(outputs["scale"], targets["scale"])
    
#     if "negative" in targets:
#         losses["negative"] = neg_loss(outputs["negative"], targets["negative"])
    
#     # gate: [batch_size, 1] -> [batch_size]
#     gate = outputs["gate"].squeeze(-1)
    
#     # Compute weighted sum of task losses: sum over tasks
#     # Each task loss is a scalar (already reduced by their respective loss functions)
#     total_raw_loss = sum(task_weights[k] * losses[k] for k in losses.keys())
    
#     # Apply gate weighting and regularization
#     # (1 - gate) * total_loss +  gate^2
#     weighted_loss = (1 - gate) * total_raw_loss + gate.pow(2)
    

#     # Normalize by task weights sum
#     total_loss = weighted_loss / sum(task_weights.values())
#     return (total_loss, losses, tag_hits_k) if hits_k else (total_loss, losses)   

# %% [markdown]
# # Train

# %%
# import wandb
# # 清 GPU
# with torch.no_grad():
#     torch.cuda.empty_cache()
# # torch.cuda.empty_cache()
# del model, input_ids, attention_mask, start_tokens, end_tokens, targets, outputs

# # 結束上次的紀錄
# wandb.finish()

# %% [markdown]
# ### Train

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


def get_gate_reg_weights(epoch):
    """
    Define per-task gate regularization weights (lambda values).
    
    Higher lambda = harder for the gate to close (more regularization)
    Lower lambda = easier for the gate to close (less regularization)
    
    Default initialization: all weights set to 1.0
    You can customize these based on task characteristics.
    
    Example interpretation:
    - tag: 1.0   -> Medium regularization (important task, selective gating)
    - time: 0.5  -> Low regularization (noisier task, can ignore samples more easily)
    - scale: 0.8 -> Medium regularization
    - negative: 1.2 -> High regularization (important for signal, less gating)
    """
    return config['training']['gate_reg_weights']


# %%
import wandb
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from transformers import get_scheduler
from tqdm import tqdm
import os

num_warmup_steps = 5
num_epochs = config['training']['num_epochs']
eval_step = config['training']['eval_step'] # 150222 個 batch
# checkpoint_save_step = 50
task_weights = config['training']['task_weights']
# task_weights = get_task_weights(0)
patience = config['training']['patience']
max_saved_models = config['training']['max_saved_models']
saved_models = [] 

model_name = config['model']['name']
date = config['model']['date']
index = config['model']['index']
run_name = f"{model_name}-{date}-{index}"

model = MultiTaskModel(
    config['model']['base_model'],
    num_tags = len(tag_list), 
    num_times = len(time_list),
    num_scales= len(scale_list)
)


bert_lr = config['training']['learning_rates']['bert']
tag_head_lr = config['training']['learning_rates']['tag_head']
time_head_lr = config['training']['learning_rates']['time_head']
scale_head_lr = config['training']['learning_rates']['scale_head']
negative_head_lr = config['training']['learning_rates']['negative_head']

optimizer = AdamW([
    {"params": model.bert.parameters(), "lr": bert_lr, "weight_decay": 1e-2},  
    {"params": model.tag_head.parameters(), "lr": tag_head_lr,  "weight_decay": 1e-2},  
    {"params": model.time_head.parameters(), "lr": time_head_lr,  "weight_decay": 1e-2},  
    {"params": model.scale_head.parameters(), "lr": scale_head_lr,  "weight_decay": 1e-2},
    {"params": model.negative_head.parameters(), "lr": negative_head_lr,  "weight_decay": 1e-2},  
])

num_total_steps= num_epochs * train_approx_batches
scheduler = CosineAnnealingLR(optimizer, T_max = num_total_steps // 4, eta_min = 1e-7)

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = 'cpu'
print(device)
best_val_loss = float("inf")
early_stop_counter = 0 


os.makedirs(f'model_weight/{model_name}', exist_ok=True)
os.makedirs(f'check_point/{model_name}', exist_ok=True)


# import wandb

# # generate a new run name or ID if needed
# run_name = f"{model_name}-{date}-{index}-gate_mechanism"

# wandb.init(
#     project="multi-task-model",
#     name=run_name,
#     # Remove 'id' and 'resume' if this is a fresh start for the new model logic
#     # id='new_id_here', 
#     # resume="allow", 
#     config={
#         "epochs": num_epochs,
#         "batch_size": batch_size,
#         "learning_rates": { 
#             "bert": bert_lr,
#             "tag_head": tag_head_lr,
#             "time_head": time_head_lr,
#             "scale_head": scale_head_lr,
#             "negative_head": negative_head_lr,
#             "gate_head": bert_lr # or whatever custom LR you used for gate
#         },
#         "task_weights": task_weights,
#         "train_data_size": train_total_samples,
#         "model_architecture": "SecBERT + GateHead + HybridLoss"
#     }
# )

# # Optional: Watch the model to see gradients (specifically helpful to see if GateHead is dying)
# wandb.watch(model, log="gradients", log_freq=100)
# # wandb.watch(model, log="all")


wandb.init(
    project = "multi-task-model",
    resume="allow",
    id='wqugd5cx',
    name = f'{run_name}_cont_0330_173',
    config = {
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
        "eval_step": eval_step,
        "model": model
    },
)


# %% [markdown]
# #### Validation

# %%
def validate_model(model, val_loader, task_weights, gate_reg_weights, device):
    model.eval()
    val_loss = 0
    valid_samples = 0
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
            values = batch["value"].to(device)
            
            targets = {
                "tag": batch["tag"].to(device),
                "time": batch["time"].to(device),
                "scale": batch["scale"].to(device),
                "negative": batch["negative"].to(device)
            }
            
            outputs = model(input_ids, attention_mask, start_tokens, end_tokens)
            loss, losses = compute_loss(outputs, targets, values, task_weights, gate_reg_weights,tag_loss_fn=tag_loss_fn, time_loss_fn=time_loss_fn, scale_loss_fn=scale_loss_fn, neg_loss=neg_loss )
            
            val_loss += loss.item()
            valid_samples += len(batch["input_ids"])
            valid_batch_count += 1
            
            for key in all_losses:
                loss_value = losses.get(key, 0)
                all_losses[key] += loss_value

    # Calculate average loss
    val_loss /= valid_batch_count
    for key in all_losses:
        all_losses[key] /= valid_batch_count
    
    return val_loss, all_losses


# %% [markdown]
# ## save func

# %%
from shared_utils import save_checkpoint, load_checkpoint


# %% [markdown]
# #### Training

# %%
# 一般的 loss with task-specific gates and gate regularization
model = model.to(device)
progress_bar = tqdm(range(num_total_steps), desc = "Training", dynamic_ncols = True)
step = 0

# Initialize gate regularization weights
gate_reg_weights = get_gate_reg_weights(0)

for epoch in range(num_epochs):
    if early_stop_counter >= patience:
        print("Early stopping triggered before starting a new epoch. Training stopped.")
        break  # 停止整個 training loop
    model.train()
    # task_weights = get_task_weights(epoch)
    # Optionally update gate_reg_weights per epoch
    # gate_reg_weights = get_gate_reg_weights(epoch)
    print(f"Task weight: {task_weights}")
    print(f"Gate Reg Weights: {gate_reg_weights}")
    # loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", total=train_approx_batches)
    train_loss = 0
    train_samples = 0
    train_batch_count = 0
    train_losses = {}
    
    for batch in train_loader:
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
        loss, losses = compute_loss(outputs, targets, values, task_weights, gate_reg_weights
        ,tag_loss_fn=tag_loss_fn,time_loss_fn=time_loss_fn,scale_loss_fn=scale_loss_fn,neg_loss=neg_loss_fn)
        # print(f'loss: {loss.item()} losses: {losses}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

       
        
        # 累計 loss
        train_loss += loss.item()
        for key, val in losses.items():
            train_losses[key] = train_losses.get(key, 0) + val
            
        train_samples += len(batch["input_ids"])
        train_batch_count += 1
        
        step += 1
        epoch_progress = step / train_approx_batches
        
        # progress_bar.set_postfix(loss = loss.item())
        progress_bar.set_postfix(
            loss = loss.item(),
            epoch = f"{epoch_progress:.2f}"
        )
        progress_bar.update(1)

        wandb.log({
            "step": step, 
            "epoch": epoch_progress,
            "train_batch_total_loss": loss.item(), # Log total loss per batch
            "lr_bert": optimizer.param_groups[0]['lr'],
            "lr_tag_head": optimizer.param_groups[1]['lr'],
            "lr_time_head": optimizer.param_groups[2]['lr'],
            "lr_scale_head": optimizer.param_groups[3]['lr'],
            "lr_negative_head": optimizer.param_groups[4]['lr'],
        }) 

    
        if step % eval_step == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, step, f'check_point/{model_name}/{date}_{index}_step{step}.pth')
            
            avg_train_loss = train_loss / train_batch_count if train_batch_count > 0 else 0
            train_loss_dict = {f"train_loss_{key}": train_losses[key] / train_batch_count for key in train_losses} if train_batch_count > 0 else {}
            
            train_loss = 0
            train_batch_count = 0
            train_losses = {}

            val_loss, val_losses = validate_model(model, valid_loader, task_weights, gate_reg_weights, device)
            val_loss_dict = {f"val_loss_{key}": val_losses[key] for key in val_losses}
            model.train()
            
            wandb.log({
                "step": step, 
                "epoch_progress": epoch_progress,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                **train_loss_dict,
                **val_loss_dict
            })
    
            print(f"\nStep {step} Epoch {epoch_progress:.2f}: \nTrain Loss = {avg_train_loss:.4f} Validation Loss = {val_loss:.4f}")
            print(f"Validation Loss Breakdown: {val_losses}")
                
            
        # print(f"\nEpoch {epoch + 1}/{num_epochs} - Training Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}")
        # print(f"Validation Loss Breakdown: {val_losses}")
        
            # Save model if validation loss improves
            if val_loss < best_val_loss:
                print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
                best_val_loss = val_loss
                early_stop_counter = 0
                
                model_save_path_epoch = f"model_weight/{model_name}/{date}_{index}_step{step+1}.pt"
                torch.save(model.state_dict(), model_save_path_epoch)
                saved_models.append(model_save_path_epoch)
                if len(saved_models) > max_saved_models:
                    oldest_model = saved_models.pop(0) 
                    if os.path.exists(oldest_model):
                        os.remove(oldest_model)
                        print(f"Removed old model: {oldest_model}")
                        
            else:
                early_stop_counter += 1
                print(f"No improvement. Early stop counter: {early_stop_counter}/{patience}")
        
            # Early stopping check
            if early_stop_counter >= patience:
                print("Early stopping triggered. Training stopped.")
                break

# 結束 wandb
wandb.finish()
 


