"""
Shared utilities for SECBERT model training and testing.
Contains common functions and classes used across secbert_smaller.ipynb and secbert_smaller_test.ipynb
"""

import random
import json
import math
import locale
import os
import csv
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from word2number import w2n
from tqdm import tqdm
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import BertModel
from sklearn.utils.class_weight import compute_class_weight

# ============================================================================
# Global Constants
# ============================================================================

CLASSIFICATION_MISSING_VALUE = -100
NUMERIC_MISSING_VALUE = torch.finfo(torch.float32).max  # 3.4028235e+38
MIN_WEIGHT = 1

# ============================================================================
# Setup & Helpers
# ============================================================================


def set_seed(seed):
    """設定隨機種子以保證可重現性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_span_to_number(span):
    """
    將 span 轉換為數字。
    """
    span = span.strip()
    
    # 嘗試直接轉換為數字
    try:
        return locale.atof(span.replace(",", ""))  # 去掉千分位逗號並轉換
    except ValueError:
        pass  # 不是標準數字，繼續嘗試解析
    
    # 嘗試將文字轉為數字
    try:
        return w2n.word_to_num(span.lower())
    except ValueError:
        pass  # 不是可解析的數字
    
    return None  # 解析失敗，返回 None


def load_counter(target_attr):
    """載入計數器資訊"""
    target_path = f'processed_iterable_dataset/counter/secbert_train_small_{target_attr}.json'
    
    with open(target_path, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data
# ============================================================================
# SelfMix Helpers
# ============================================================================

def compute_multitask_kl_loss(logits_1, logits_2, ignore_index=-100):
    """
    Computes symmetric KL divergence across multiple task logits.
    Ensures consistency between two dropout-perturbed forward passes.
    """
    total_kl = 0.0
    valid_tasks = 0
    
    for task_key in logits_1.keys():
        p = logits_1[task_key]
        q = logits_2[task_key]
        
        # Symmetrical KL Divergence
        p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='batchmean')
        q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='batchmean')
        
        total_kl += (p_loss + q_loss) / 2
        valid_tasks += 1
        
    return total_kl / valid_tasks if valid_tasks > 0 else torch.tensor(0.0).to(logits_1['tag'].device)

# ============================================================================
# Data Processing
# ============================================================================


def process_batch(batch, target_attrs, tokenizer, tag2id=None, time2id=None, scale2id=None, standard_rare_tags=None):
    """
    處理一個批次的數據。
    
    Args:
        batch: 批次數據
        target_attrs: 目標屬性列表
        tokenizer: 分詞器
        tag2id: tag 到 id 的映射
        time2id: time 到 id 的映射
        scale2id: scale 到 id 的映射
        standard_rare_tags: 罕見標籤集合
    """
    
    batch_results = []
    for job in batch:
        context_p = job['context'].get("context_p", "")
        context_t = job['context'].get("context_t", "")
        context_n = job['context'].get("context_n", "")
        
        document_info = f"{job['document']['document_type']};{job['document']['period_end_date']};{job['document']['fiscal_year']};{job['document']['period_focus']}"
        full_context = f"{context_t} [SEP] {document_info} [SEP] {context_p} [SEP] {context_n}"

        tokenized = tokenizer(
            full_context,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        token_ids = tokenized["input_ids"].squeeze(0)
        offset_mapping = tokenized["offset_mapping"].squeeze(0)

        for i, target in enumerate(job['targets']):
            start_char, end_char = target['start_pos'], target['end_pos']
            start_token, end_token = -1, -1

            for idx, (start, end) in enumerate(offset_mapping):
                if start <= start_char < end:
                    start_token = idx
                if start < end_char <= end:
                    end_token = idx
                    break
            
            if start_token == -1 or end_token == -1:
                continue 
                
            target_data = {
                "job_id": job["job_id"],
                "seq_id": target["seq_id"],
                "context": full_context,
                "input_ids": token_ids,
                "attention_mask": tokenized['attention_mask'].squeeze(0),
                "start_token": start_token,
                "end_token": end_token,
                "value": convert_span_to_number(target['text']),
                "doc_link": job['document']['document_link'],
            }
            
            for attr in target_attrs:
                if attr in ["tag", "time", "scale", "negative"]:  # 分類屬性
                    target_data[attr] = CLASSIFICATION_MISSING_VALUE
                elif attr == "fact":  # 數值屬性
                    target_data[attr] = NUMERIC_MISSING_VALUE

            gold_values = job['golds'][i]['value']
            for attr_idx, attr in enumerate(target['attribute']):
                value = gold_values[attr_idx]
                if attr == 'tag':
                    if standard_rare_tags and value in standard_rare_tags:
                        value = 'standard_rare'
                    target_data['tag'] = tag2id.get(value, -100) if tag2id else -100
                elif attr == 'time':
                    target_data['time'] = time2id.get(value, -100) if time2id else -100
                elif attr == 'fact':
                    if value:
                        target_data['fact'] = float(value)
                        target_data['negative'] = 1 if value < 0 else 0
                    else:
                        target_data['fact'] = NUMERIC_MISSING_VALUE
                        target_data['negative'] = CLASSIFICATION_MISSING_VALUE
                elif attr == 'scale':
                    target_data['scale'] = scale2id.get(value, -100) if scale2id else -100

            batch_results.append(target_data)
    return batch_results


def process_batch_wrapper(args):
    """用於 `ThreadPoolExecutor` 的批次處理函數"""
    batch, target_attrs, tokenizer, tag2id, time2id, scale2id, standard_rare_tags = args
    return process_batch(batch, target_attrs, tokenizer, tag2id, time2id, scale2id, standard_rare_tags)


def process_data(data, target_attrs, tokenizer, batch_size=32, num_workers=8, tag2id=None, time2id=None, scale2id=None, standard_rare_tags=None):
    """
    使用多進程處理數據批次。
    """
    inputs = []
    
    # 計算總批次數
    num_batches = math.ceil(len(data) / batch_size)

    # 將數據拆分成批次
    batches = [data[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

    # 構建參數列表
    task_args = [(batch, target_attrs, tokenizer, tag2id, time2id, scale2id, standard_rare_tags) for batch in batches]

    # 使用多進程處理批次
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_batch_wrapper, task_args), total=num_batches, desc="Processing Data"))

    # 合併所有批次的結果
    for res in results:
        inputs.extend(res)
    
    return inputs


# ============================================================================
# IterableDataset Classes
# ============================================================================


class MultiTaskIterableDataset(IterableDataset):
    """讀取原始 JSONL 檔案，進行前處理"""

    def __init__(self, files, target_attrs, tokenizer, batch_size=32, num_workers=8, tag2id=None, time2id=None, scale2id=None, standard_rare_tags=None):
        self.files = files
        self.target_attrs = target_attrs
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tag2id = tag2id
        self.time2id = time2id
        self.scale2id = scale2id
        self.standard_rare_tags = standard_rare_tags
    
    def _get_sharded_lines(self, file_path):
        """確保多進程時，每個 worker 讀不同的部分"""
        worker_info = get_worker_info()
        if worker_info is None:  # 單進程模式
            start, step = 0, 1
        else:
            start, step = worker_info.id, worker_info.num_workers  # worker id & 總數

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % step == start:  # 讓不同 worker 讀不同的行
                    yield line
    
    def __iter__(self):
        """逐行讀取 JSONL 並轉換為數據格式"""
        for file_path in self.files:
            for line in self._get_sharded_lines(file_path):
                raw_data = json.loads(line)
                processed_data = process_batch([raw_data], self.target_attrs, self.tokenizer, self.tag2id, self.time2id, self.scale2id, self.standard_rare_tags)
                for item in processed_data:
                    yield item


class BufferedShuffleDataset(IterableDataset):
    """負責對 IterableDataset 進行 buffer shuffle"""

    def __init__(self, dataset, buffer_size=8000):
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        buffer = []
        for sample in self.dataset:
            buffer.append(sample)
            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)
                while buffer:
                    yield buffer.pop()
        
        random.shuffle(buffer)
        while buffer:
            yield buffer.pop()


class ProcessedIterableDataset(IterableDataset):
    """讀取處理後的 JSONL"""
    
    def __init__(self, files):
        self.files = files

    def _get_sharded_lines(self, file_path):
        """確保多進程時，每個 worker 讀不同的部分"""
        worker_info = get_worker_info()
        if worker_info is None:  # 單進程模式
            start, step = 0, 1
        else:
            start, step = worker_info.id, worker_info.num_workers  # worker id & 總數

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % step == start:  # 讓不同 worker 讀不同的行
                    yield line

    def __iter__(self):
        """讀取 JSONL 並轉換為合適格式"""
        for file_path in self.files:
            for line in self._get_sharded_lines(file_path):
                raw_data = json.loads(line)

                # 把 list 轉回 torch.Tensor
                processed_data = {
                    key: torch.tensor(value) if isinstance(value, list) else value
                    for key, value in raw_data.items()
                }
                
                yield processed_data

# ============================================================================
# Model
# ============================================================================

class MultiTaskModel(nn.Module):
    """Multi-task BERT model refactored for SelfMix textual-level mixup."""
    
    def __init__(self, bert_model_name, num_tags, num_times, num_scales):
        super(MultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        
        self.tag_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2), 
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_tags)
        )

        self.time_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2), 
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, num_times)
        )

        self.scale_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2), 
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, num_scales)
        )

        self.negative_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2), 
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, 2)
        )
        
    def get_embeddings(self, input_ids, attention_mask, start_tokens, end_tokens):
        """Extracts text-level embeddings for target spans, enabling Mixup."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Aggregate target embeddings based on span positions
        target_embeddings = [
            sequence_output[i, start_tokens[i]:end_tokens[i] + 1].mean(dim=0)
            for i in range(input_ids.size(0))
        ]
        return torch.stack(target_embeddings)

    def classify(self, target_embeddings):
        """Passes aggregated span embeddings through task-specific MLPs."""
        tag_logits = self.tag_head(target_embeddings)
        time_logits = self.time_head(target_embeddings)
        scale_logits = self.scale_head(target_embeddings)
        negative_logits = self.negative_head(target_embeddings)
        
        return {
            "tag": tag_logits,
            "time": time_logits,
            "scale": scale_logits,
            "negative": negative_logits,
        }

    def forward(self, input_ids, attention_mask, start_tokens, end_tokens):
        target_embeddings = self.get_embeddings(input_ids, attention_mask, start_tokens, end_tokens)
        return self.classify(target_embeddings)

# ============================================================================
# Metrics & Loss
# ============================================================================


def hits_at_k(predictions, targets, k=5):
    """
    計算 Hits@K 指標
    """
    valid_mask = targets != CLASSIFICATION_MISSING_VALUE
    targets = targets[valid_mask]
    predictions = predictions[valid_mask]

    top_k_preds = torch.topk(predictions, k, dim=-1).indices

    if targets.dim() == 1:
        targets = torch.nn.functional.one_hot(targets, num_classes=predictions.size(1))

    targets = targets.float()

    hits = torch.any(targets.gather(1, top_k_preds), dim=1).float()

    return hits.mean().item()


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean", ignore_index=CLASSIFICATION_MISSING_VALUE):
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([1 - alpha, alpha])
        else:
            self.alpha = torch.tensor(alpha) 
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        device = logits.device
        self.alpha = self.alpha.to(device)
        
        if targets.dim() > 1:
            targets = targets.argmax(dim=-1)
        
        valid_mask = (targets != self.ignore_index)
        targets = targets[valid_mask]
        logits = logits[valid_mask]

        ce_loss = self.ce_loss(logits, targets)
        pt = torch.exp(-ce_loss)
        
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = self.alpha.gather(0, targets.data.view(-1))

        loss = alpha_weight * focal_weight * ce_loss

        if self.reduction == "mean":
            return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=device, requires_grad=True)
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# class CB_CE_Loss(nn.Module):
#     """Class-Balanced Cross Entropy Loss"""
    
#     def __init__(self, num_samples, beta=0.99, ignore_index=CLASSIFICATION_MISSING_VALUE):
#         super(CB_CE_Loss, self).__init__()
        
#         effective_num = 1.0 - torch.pow(torch.tensor(beta), torch.tensor(num_samples))
#         weights = (1.0 - beta) / (effective_num + 1e-8)
#         self.weights = weights
#         self.ignore_index = ignore_index
        
#     def forward(self, logits, targets):
#         if targets.dim() > 1:
#             targets = targets.argmax(dim=-1)
            
#         valid_mask = (targets != self.ignore_index)
#         targets = targets[valid_mask]
#         logits = logits[valid_mask]
        
#         ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        
#         class_weights = self.weights.to(logits.device)
#         weighted_loss = ce_loss * class_weights[targets]
        
#         return torch.mean(weighted_loss)

# class CB_CE_Loss(nn.Module):
#     def __init__(self, weights, ignore_index=-100):
#         super().__init__()
#         # 1. Ensure weights are a Tensor
#         if not isinstance(weights, torch.Tensor):
#             self.weights = torch.tensor(weights, dtype=torch.float)
#         else:
#             self.weights = weights
            
#         self.ignore_index = ignore_index

#     def forward(self, logits, targets):
#         # Calculate standard CE loss
#         ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        
#         class_weights = self.weights.to(logits.device)
#         num_weights = len(class_weights)
        
#         # 2. Create a safe indices tensor
#         safe_targets = targets.clone()
        
#         # Identify invalid indices:
#         # - The specific ignore_index (-100)
#         # - Any index larger than the number of weights we have
#         # - Any negative index (other than ignore_index)
#         invalid_mask = (safe_targets == self.ignore_index) | (safe_targets >= num_weights) | (safe_targets < 0)
        
#         # Replace invalid indices with 0 to prevent "Index Out of Bounds" crash
#         safe_targets[invalid_mask] = 0
        
#         # 3. Get weights
#         sample_weights = class_weights[safe_targets]
        
#         # 4. Zero out weights for invalid entries so they don't affect loss
#         sample_weights[invalid_mask] = 0.0
        
#         # Apply weights
#         weighted_loss = ce_loss * sample_weights

#         return torch.mean(weighted_loss)

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


class CB_CE_Loss(nn.Module):
    '''
    https://ieeexplore.ieee.org/abstract/document/8953804
    '''
    def __init__(self, num_samples, beta=0.99, ignore_index=CLASSIFICATION_MISSING_VALUE):
        """
        Args:
            num_samples: list or tensor, 每個類別的樣本數
            beta: 控制 class-balanced 權重的超參數 (通常取 0.99)
        """
        super(CB_CE_Loss, self).__init__()
        
        # 計算 Class-Balanced 權重
        effective_num = 1.0 - torch.pow(torch.tensor(beta), torch.tensor(num_samples))
        weights = (1.0 - beta) / (effective_num + 1e-8)
        # self.weights = weights / torch.sum(weights)  # normalize
        self.weights = weights
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes) 模型輸出的 logits
            targets: (batch_size,) 類別索引標籤
        Returns:
            CB-CE Loss 值
        """
        
        if targets.dim() > 1:
            targets = targets.argmax(dim=-1)
            
        valid_mask = (targets != self.ignore_index)  # 只對有效的 targets 計算 loss
        targets = targets[valid_mask]
        logits = logits[valid_mask]
        
        # 計算標準 CE Loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        
        # 依照類別權重調整 loss
        class_weights = self.weights.to(logits.device)
        weighted_loss = ce_loss * class_weights[targets]
        
        # weighted_loss = ce_loss * class_weights[targets] * weight_mask.float()
        return torch.mean(weighted_loss)

        # return weighted_loss.sum() / weight_mask.sum()  # 只對有效樣本取平均




def huber_loss(y_true, y_pred, delta=1.0):
    """Huber Loss"""
    error = y_true - y_pred
    is_small_error = torch.abs(error) < delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    return torch.where(is_small_error, squared_loss, linear_loss).mean()


def signed_log(x):
    """Signed logarithm"""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def fact_loss_fn(fact_pred, fact_target, mse_loss=None):
    """計算事實預測的損失"""
    if mse_loss is None:
        mse_loss = nn.MSELoss()
    
    valid_mask = fact_target != NUMERIC_MISSING_VALUE
    fact_pred = fact_pred[valid_mask]
    fact_target = fact_target[valid_mask]
    
    fact_target_log = signed_log(fact_target)
    fact_pred_log = signed_log(fact_pred)

    threshold = 8.0
    use_huber = fact_target_log.abs() < threshold
    use_mse = ~use_huber

    huber_part = huber_loss(fact_pred_log[use_huber], fact_target_log[use_huber]) if use_huber.any() else 0
    mse_part = mse_loss(fact_pred_log[use_mse], fact_target_log[use_mse]) if use_mse.any() else 0

    return huber_part + mse_part


def compute_loss(outputs, targets, values, task_weights=None, hits_k=False, 
                 tag_loss_fn=None, time_loss_fn=None, scale_loss_fn=None, 
                 neg_loss_fn=None, classification_loss=None, mse_loss=None):
    """
    計算多任務損失。
    
    Args:
        outputs: 模型輸出
        targets: 目標值
        values: 數值目標
        task_weights: 任務權重
        hits_k: 是否計算 hits@k
        tag_loss_fn, time_loss_fn, scale_loss_fn, neg_loss: 各任務的損失函數
        classification_loss, mse_loss: 通用損失函數
    """
    
    if task_weights is None:
        task_weights = {"tag": 1.0, "time": 1.0, "scale": 1.0, "negative": 1.0}
    
    if classification_loss is None:
        classification_loss = nn.CrossEntropyLoss(ignore_index=CLASSIFICATION_MISSING_VALUE)
    
    if mse_loss is None:
        mse_loss = nn.MSELoss()
    
    losses = {}
    tag_hits_k = {}
    
    if "tag" in targets and tag_loss_fn is not None:
        losses["tag"] = tag_loss_fn(outputs["tag"], targets["tag"])
        
        if hits_k:
            tag_hits_k["hits_1"] = hits_at_k(outputs["tag"], targets["tag"], k=1)
            tag_hits_k["hits_3"] = hits_at_k(outputs["tag"], targets["tag"], k=3)
            tag_hits_k["hits_5"] = hits_at_k(outputs["tag"], targets["tag"], k=5)

    if "time" in targets and time_loss_fn is not None:
        losses["time"] = time_loss_fn(outputs["time"], targets["time"])
    
    if "scale" in targets and scale_loss_fn is not None:
        losses["scale"] = scale_loss_fn(outputs["scale"], targets["scale"])
    
    if "negative" in targets and neg_loss_fn is not None:
        losses["negative"] = neg_loss_fn(outputs["negative"], targets["negative"])
    
    total_loss = sum(task_weights.get(k, 1.0) * losses[k] for k in losses.keys())
    
    total_loss = total_loss / sum(task_weights.values())

    return (total_loss, losses, tag_hits_k) if hits_k else (total_loss, losses)


# ============================================================================
# Checkpoint Management
# ============================================================================


def save_checkpoint(model, optimizer, scheduler, epoch, step, save_path="checkpoint.pth"):
    """保存檢查點"""
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved at {save_path}")


def load_checkpoint(model, optimizer, scheduler, save_path, device):
    """載入檢查點"""
    checkpoint = torch.load(save_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    print(f"Checkpoint loaded: epoch {epoch}, step {step}")
    return epoch, step


# ============================================================================
# Validation & Evaluation
# ============================================================================


def validate_model(model, val_loader, task_weights, device, 
                   tag_loss_fn=None, time_loss_fn=None, scale_loss_fn=None, neg_loss_fn=None):
    """驗證模型"""
    model.eval()
    val_loss = 0
    valid_samples = 0
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
            valid_samples += len(batch["input_ids"])
            valid_batch_count += 1
            
            for key in all_losses:
                loss_value = losses.get(key, 0)
                all_losses[key] += loss_value if isinstance(loss_value, (int, float)) else loss_value.item()

    # Calculate average loss
    val_loss /= valid_batch_count
    for key in all_losses:
        all_losses[key] /= valid_batch_count
    
    return val_loss, all_losses


def evaluate_model(model, model_save_path, error_file_path, test_loader, task_weights, device, 
                   id2scale=None, save_errors=False):
    """評估測試集上的性能"""
    os.environ["WANDB_DISABLED"] = "true"
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    all_losses = {'tag': 0, 'time': 0, 'fact': 0, 'scale': 0, 'negative': 0}
    total_hits = {"hits_1": 0, "hits_3": 0, "hits_5": 0}
    predictions = defaultdict(list)
    ground_truths = defaultdict(list)
    errors = defaultdict(list)
    test_samples = 0
    test_batch_count = 0
    
    fact_res = []
    classification_loss = nn.CrossEntropyLoss(ignore_index=CLASSIFICATION_MISSING_VALUE)
    
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
            loss, losses, tag_hits_k = compute_loss(outputs, targets, values, task_weights, hits_k=True)
            test_samples += len(batch["input_ids"])
            test_batch_count += 1
            
            if tag_hits_k:
                for key in total_hits:
                    total_hits[key] += tag_hits_k[key]
                    
            for key in all_losses:
                if key in losses:
                    all_losses[key] += losses[key].item() if hasattr(losses[key], 'item') else losses[key]
                
            for key in ["scale", "negative", "tag", "time"]:
                pred = torch.argmax(outputs[key], dim=-1).cpu().tolist()
                true = targets[key].cpu().tolist()

                for i, (p, t) in enumerate(zip(pred, true)):
                    predictions[key].append(p)
                    ground_truths[key].append(t)
                    if p != t:
                        errors[key].append({
                            "batch_idx": batch_idx,
                            "sample_idx": i,
                            "true": t,
                            "pred": p,
                        })

            # Handle fact predictions
            if "fact" in targets:
                negative_pred = outputs["negative"].argmax(dim=-1).cpu().numpy()
                losses["negative"] = classification_loss(outputs["negative"], targets["negative"])
                
                if "scale" in outputs:
                    scale_pred_class = outputs["scale"].argmax(dim=-1).cpu().numpy()
                else:
                    scale_pred_class = np.ones_like(values.cpu().numpy(), dtype=np.int64)
            
                if id2scale is not None:
                    id2scale_np = np.array([float(id2scale[idx]) for idx in range(len(id2scale))], dtype=np.float64)
                    scale_pred_values = id2scale_np[scale_pred_class]
                else:
                    scale_pred_values = np.zeros_like(scale_pred_class, dtype=np.float64)

                values = values.cpu().numpy()
                fact_pred = values * (-1) ** negative_pred * (10 ** scale_pred_values)
                fact_target = targets["fact"].view(-1).cpu().numpy()

                for i, (p, t, neg, scale, scale_val, val) in enumerate(zip(fact_pred, fact_target, negative_pred, scale_pred_class, scale_pred_values, values)):
                    if t == NUMERIC_MISSING_VALUE:
                        continue
                
                    predictions["fact"].append(p)
                    ground_truths["fact"].append(t)
                    
                    fact_res.append({
                        "batch_idx": batch_idx,
                        "sample_idx": i,
                        "true": t,
                        "pred": p,
                        "negative_pred": neg,
                        "scale_pred": scale,
                        "scale_val": scale_val,
                        "value": val
                    })
                    if abs(p - t) > 1e-2:
                        errors["fact"].append({
                            "batch_idx": batch_idx,
                            "sample_idx": i,
                            "true": t,
                            "pred": p,
                        })

    for key in all_losses:
        all_losses[key] /= test_batch_count
    
    avg_hits_k = {key: total_hits[key] / test_batch_count for key in total_hits}
        

    # Save errors to CSV
    if save_errors:
        for key, error_list in errors.items():
            error_file = f"{error_file_path}_errors_{key}.csv"
            with open(error_file, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["batch_idx", "sample_idx", "true", "pred"])
                writer.writeheader()
                writer.writerows(error_list)
            print(f"🚨 錯誤樣本已儲存到 {error_file}")

    if fact_res:
        fact_res_df = pd.DataFrame(fact_res)
        fact_res_df.to_csv('fact_res.csv', index=False)
    
    return all_losses, dict(predictions), dict(ground_truths), avg_hits_k
