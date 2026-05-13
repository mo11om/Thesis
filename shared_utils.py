"""
Shared utilities for XBRL multi-task model training.
Contains dataset processing, model architectures, and loss functions.
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import locale
import os
import math
import csv
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from word2number import w2n
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import BertModel
from sklearn.metrics import (
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from collections import defaultdict


# ============================================================================
# initialization

# ===========================================================================




# ============================================================================
# Seed & Constants
# ============================================================================

def set_seed(seed):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# Data Processing Functions
# ============================================================================

def convert_span_to_number(span):
    """
    Convert span text to numeric value.
    Attempts direct conversion first, then word-to-number conversion.
    
    Args:
        span: String containing numeric value or word representation
        
    Returns:
        float: Numeric value, or None if conversion fails
    """
    span = span.strip()
    
    # Try direct conversion
    try:
        return locale.atof(span.replace(",", ""))
    except ValueError:
        pass
    
    # Try word-to-number conversion
    try:
        return w2n.word_to_num(span.lower())
    except ValueError:
        pass
    
    return None


def process_batch(batch, target_attrs, tokenizer, standard_rare_tags, tag2id, 
                  time2id, scale2id, CLASSIFICATION_MISSING_VALUE, NUMERIC_MISSING_VALUE):
    """
    Process a batch of raw JSONL data into tokenized training samples.
    
    Args:
        batch: List of job dictionaries from JSONL
        target_attrs: List of attributes to extract
        tokenizer: HuggingFace tokenizer
        standard_rare_tags: Set of rare tags to collapse
        tag2id: Dictionary mapping tags to indices
        time2id: Dictionary mapping time values to indices
        scale2id: Dictionary mapping scale values to indices
        CLASSIFICATION_MISSING_VALUE: Sentinel value for missing classifications
        NUMERIC_MISSING_VALUE: Sentinel value for missing numeric values
        
    Returns:
        List of processed target samples with tokenized text
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
                if attr in ["tag", "time", "scale", "negative"]:
                    target_data[attr] = CLASSIFICATION_MISSING_VALUE
                elif attr == "fact":
                    target_data[attr] = NUMERIC_MISSING_VALUE

            gold_values = job['golds'][i]['value']
            for attr_idx, attr in enumerate(target['attribute']):
                value = gold_values[attr_idx]
                if attr == 'tag':
                    if value in standard_rare_tags:
                        value = 'standard_rare'
                    target_data['tag'] = tag2id.get(value, -100)
                elif attr == 'time':
                    target_data['time'] = time2id.get(value, -100)
                elif attr == 'fact':
                    if value:
                        target_data['fact'] = float(value)
                        target_data['negative'] = 1 if value < 0 else 0
                    else:
                        target_data['fact'] = NUMERIC_MISSING_VALUE
                        target_data['negative'] = CLASSIFICATION_MISSING_VALUE
                elif attr == 'scale':
                    target_data['scale'] = scale2id.get(value, -100)
                    
            batch_results.append(target_data)
    return batch_results


def process_batch_wrapper(args):
    """Wrapper function for ProcessPoolExecutor batch processing."""
    batch, target_attrs, tokenizer, standard_rare_tags, tag2id, time2id, scale2id, CLASSIFICATION_MISSING_VALUE, NUMERIC_MISSING_VALUE = args
    return process_batch(batch, target_attrs, tokenizer, standard_rare_tags, tag2id, 
                         time2id, scale2id, CLASSIFICATION_MISSING_VALUE, NUMERIC_MISSING_VALUE)


def process_data(data, target_attrs, tokenizer, standard_rare_tags, tag2id, 
                 time2id, scale2id, CLASSIFICATION_MISSING_VALUE, NUMERIC_MISSING_VALUE,
                 batch_size=32, num_workers=8):
    """
    Process data using multi-threaded batch processing.
    
    Args:
        data: List of raw job dictionaries
        target_attrs: Attributes to extract
        tokenizer: HuggingFace tokenizer
        standard_rare_tags: Set of rare tags
        tag2id: Tag to index mapping
        time2id: Time to index mapping
        scale2id: Scale to index mapping
        CLASSIFICATION_MISSING_VALUE: Missing value sentinel
        NUMERIC_MISSING_VALUE: Numeric missing value sentinel
        batch_size: Batch size for processing
        num_workers: Number of worker threads
        
    Returns:
        List of processed samples
    """
    inputs = []
    num_batches = math.ceil(len(data) / batch_size)
    batches = [data[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]
    
    task_args = [(batch, target_attrs, tokenizer, standard_rare_tags, tag2id, time2id, 
                  scale2id, CLASSIFICATION_MISSING_VALUE, NUMERIC_MISSING_VALUE) for batch in batches]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_batch_wrapper, task_args), total=num_batches, desc="Processing Data"))

    for res in results:
        inputs.extend(res)
    
    return inputs


def count_lines(file_path):
    """Count number of lines in a file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_counter(target_attr, counter_dir='processed_iterable_dataset/counter'):
    """Load class count statistics from JSON file."""
    target_path = f'{counter_dir}/secbert_train_small_{target_attr}.json'
    with open(target_path, "r", encoding='utf-8') as f:
        return json.load(f)


# ============================================================================
# Dataset Classes
# ============================================================================

class MultiTaskIterableDataset(IterableDataset):
    """
    IterableDataset that reads raw JSONL files and processes them on-the-fly.
    Supports multi-process data loading with proper sharding.
    """

    def __init__(self, files, target_attrs, tokenizer, standard_rare_tags, tag2id, 
                 time2id, scale2id, CLASSIFICATION_MISSING_VALUE, NUMERIC_MISSING_VALUE,
                 batch_size=32, num_workers=8):
        self.files = files
        self.target_attrs = target_attrs
        self.tokenizer = tokenizer
        self.standard_rare_tags = standard_rare_tags
        self.tag2id = tag2id
        self.time2id = time2id
        self.scale2id = scale2id
        self.CLASSIFICATION_MISSING_VALUE = CLASSIFICATION_MISSING_VALUE
        self.NUMERIC_MISSING_VALUE = NUMERIC_MISSING_VALUE
        self.batch_size = batch_size
        self.num_workers = num_workers
    
    def _get_sharded_lines(self, file_path):
        """Ensure each worker reads different lines in multi-process mode."""
        worker_info = get_worker_info()
        if worker_info is None:
            start, step = 0, 1
        else:
            start, step = worker_info.id, worker_info.num_workers

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % step == start:
                    yield line
    
    def __iter__(self):
        """Read JSONL line by line and yield processed samples."""
        for file_path in self.files:
            for line in self._get_sharded_lines(file_path):
                raw_data = json.loads(line)
                processed_data = process_batch([raw_data], self.target_attrs, self.tokenizer,
                                                self.standard_rare_tags, self.tag2id, self.time2id,
                                                self.scale2id, self.CLASSIFICATION_MISSING_VALUE, 
                                                self.NUMERIC_MISSING_VALUE)
                for item in processed_data:
                    yield item


class BufferedShuffleDataset(IterableDataset):
    """Provides shuffling for IterableDataset using a buffer."""

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
    """Read pre-processed JSONL files and convert lists back to tensors."""

    def __init__(self, files):
        self.files = files

    def _get_sharded_lines(self, file_path):
        """Ensure each worker reads different lines in multi-process mode."""
        worker_info = get_worker_info()
        if worker_info is None:
            start, step = 0, 1
        else:
            start, step = worker_info.id, worker_info.num_workers

        with open(file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % step == start:
                    yield line

    def __iter__(self):
        """Read JSONL and convert to tensor format."""
        for file_path in self.files:
            for line in self._get_sharded_lines(file_path):
                raw_data = json.loads(line)
                processed_data = {
                    key: torch.tensor(value) if isinstance(value, list) else value
                    for key, value in raw_data.items()
                }
                yield processed_data


# ============================================================================
# Model Classes
# ============================================================================

class GateHead(nn.Module):
    """
    Multi-Task Gate Head that outputs separate noise scores for each task.
    
    Each gate represents the "noise probability":
        1 = High Noise (Ignore sample for this task)
        0 = Clean Data (Learn from sample for this task)
    """
    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
        self.gate_tag = nn.Linear(hidden_size, 1)
        self.gate_time = nn.Linear(hidden_size, 1)
        self.gate_scale = nn.Linear(hidden_size, 1)
        self.gate_negative = nn.Linear(hidden_size, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [CLS] token representation, shape (batch_size, hidden_size)
            
        Returns:
            Dictionary with per-task gate scores, each shape (batch_size, 1)
        """
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        
        gate_tag = self.sigmoid(self.gate_tag(x))
        gate_time = self.sigmoid(self.gate_time(x))
        gate_scale = self.sigmoid(self.gate_scale(x))
        gate_negative = self.sigmoid(self.gate_negative(x))
        
        return {
            "tag": gate_tag,
            "time": gate_time,
            "scale": gate_scale,
            "negative": gate_negative
        }


class MultiTaskModel(nn.Module):
    """
    Multi-task BERT model with task-specific heads and gating mechanism.
    Performs multi-task learning on tag, time, scale, and negative predictions.
    """
    def __init__(self, bert_model_name, num_tags, num_times, num_scales):
        super().__init__()
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
        
        self.gate_head = GateHead(hidden_size)
        
    def forward(self, input_ids, attention_mask, start_tokens, end_tokens):
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            start_tokens: (batch_size,) - start indices of target span
            end_tokens: (batch_size,) - end indices of target span
            
        Returns:
            Dictionary with task predictions and per-task gates
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_token = sequence_output[:, 0, :]
        
        target_embeddings = [
            sequence_output[i, start_tokens[i]:end_tokens[i] + 1].mean(dim=0)
            for i in range(input_ids.size(0))
        ]
        target_embeddings = torch.stack(target_embeddings)
        
        tag_logits = self.tag_head(target_embeddings)
        time_logits = self.time_head(target_embeddings)
        scale_logits = self.scale_head(target_embeddings)
        negative_logits = self.negative_head(target_embeddings)
        
        gates = self.gate_head(cls_token)
        
        return {
            "tag": tag_logits,
            "time": time_logits,
            "scale": scale_logits,
            "negative": negative_logits,
            "gates": gates
        }


# ============================================================================
# Loss Functions & Metrics
# ============================================================================

def hits_at_k(predictions, targets, k=5):
    """
    Compute Hits@K metric.
    
    Args:
        predictions: (batch_size, num_tags) - prediction scores
        targets: (batch_size,) or (batch_size, num_tags) - target labels
        k: Number of top predictions to consider
        
    Returns:
        float: Average Hits@K score
    """
    valid_mask = targets != -100
    targets = targets[valid_mask]
    predictions = predictions[valid_mask]

    top_k_preds = torch.topk(predictions, k, dim=-1).indices

    if targets.dim() == 1:
        targets = torch.nn.functional.one_hot(targets, num_classes=predictions.size(1))

    targets = targets.float()
    hits = torch.any(targets.gather(1, top_k_preds), dim=1).float()

    return hits.mean().item()


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    Reference: https://doi.org/10.1109/tpami.2018.2858826
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="none", ignore_index=-100):
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
            
        ce_loss = self.ce_loss(logits, targets) 
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        safe_targets = targets.clone()
        safe_targets[safe_targets == self.ignore_index] = 0 
        alpha_weight = self.alpha.gather(0, safe_targets.view(-1))
        
        loss = alpha_weight * focal_weight * ce_loss
        loss = torch.where(targets != self.ignore_index, loss, torch.zeros_like(loss))

        if self.reduction == "mean":
            return loss.sum() / (targets != self.ignore_index).sum()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CB_CE_Loss(nn.Module):
    """
    Class-Balanced Cross Entropy Loss for imbalanced datasets.
    Reference: https://ieeexplore.ieee.org/abstract/document/8953804
    """
    def __init__(self, num_samples, beta=0.99, ignore_index=-100):
        super(CB_CE_Loss, self).__init__()
        effective_num = 1.0 - torch.pow(torch.tensor(beta), torch.tensor(num_samples))
        weights = (1.0 - beta) / (effective_num + 1e-8)
        self.weights = weights
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets):
        if targets.dim() > 1:
            targets = targets.argmax(dim=-1)
            
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        
        class_weights = self.weights.to(logits.device)
        safe_targets = targets.clone()
        safe_targets[safe_targets == self.ignore_index] = 0
        sample_weights = class_weights[safe_targets]
        
        weighted_loss = ce_loss * sample_weights
        weighted_loss = torch.where(targets != self.ignore_index, weighted_loss, torch.zeros_like(weighted_loss))
        
        return weighted_loss


def huber_loss(y_true, y_pred, delta=1.0):
    """Compute Huber loss."""
    error = y_true - y_pred
    is_small_error = torch.abs(error) < delta
    squared_loss = 0.5 * error ** 2
    linear_loss = delta * (torch.abs(error) - 0.5 * delta)
    return torch.where(is_small_error, squared_loss, linear_loss).mean()


def signed_log(x):
    """Compute signed logarithm preserving sign."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def compute_loss(outputs, targets, values, task_weights=None, gate_reg_weights=None, 
                 hits_k=False, tag_loss_fn=None, time_loss_fn=None, scale_loss_fn=None, 
                 neg_loss=None, NUMERIC_MISSING_VALUE=None, mse_loss=None):
    """
    Compute loss with task-specific gating and regularization.
    
    Formula for each task:
        L_task_final = (1 - g_task) * L_task_raw + lambda_task * (g_task)^2
    
    Total Loss = sum(task_weights[task] * L_task_final for all tasks)
    
    Args:
        outputs: Dict from model forward pass with 'gates' key
        targets: Dict of target tensors
        values: Tensor of numeric values
        task_weights: Dict mapping task names to importance weights
        gate_reg_weights: Dict mapping task names to gate regularization lambdas
        hits_k: Whether to compute Hits@K metrics
        tag_loss_fn: Loss function for tag task
        time_loss_fn: Loss function for time task
        scale_loss_fn: Loss function for scale task
        neg_loss: Loss function for negative task
        NUMERIC_MISSING_VALUE: Sentinel for numeric missing values
        mse_loss: MSE loss function
        
    Returns:
        (total_loss, losses_log) if hits_k=False
        (total_loss, losses_log, tag_hits_k) if hits_k=True
    """
 
    if task_weights is None:
        task_weights = {"tag": 1.0, "time": 1.0, "scale": 1.0, "negative": 1.0}
    if gate_reg_weights is None:
        gate_reg_weights = {"tag": 1.0, "time": 1.0, "scale": 1.0, "negative": 1.0}
    
    losses = {}
    
    if "tag" in targets and tag_loss_fn:
        losses["tag"] = tag_loss_fn(outputs["tag"], targets["tag"])

    if "time" in targets and time_loss_fn:
        losses["time"] = time_loss_fn(outputs["time"], targets["time"])
    
    if "scale" in targets and scale_loss_fn:
        losses["scale"] = scale_loss_fn(outputs["scale"], targets["scale"])
    
    if "negative" in targets and neg_loss:
        losses["negative"] = neg_loss(outputs["negative"], targets["negative"])
    
    gates = outputs.get("gates", {})
    
    device = list(losses.values())[0].device if losses else torch.device('cpu')
    loss_size = losses[list(losses.keys())[0]].shape[0] if losses else 1
    total_loss_vec = torch.zeros(loss_size, device=device)
    
    for task_name, raw_loss in losses.items():
        gate = gates[task_name].squeeze(-1)
        lambda_task = gate_reg_weights.get(task_name, 1.0)
        task_weight = task_weights.get(task_name, 1.0)
        
        gated_loss = (1 - gate) * raw_loss + lambda_task * gate.pow(2)
        total_loss_vec += task_weight * gated_loss
    
    total_loss = total_loss_vec.mean()
    
    losses_log = {k: v.mean().item() for k, v in losses.items()}
    
    for task_name, gate in gates.items():
        gate_mean = gate.mean().item()
        gate_std = gate.std().item() if gate.numel() > 1 else 0.0
        losses_log[f'gate_{task_name}_mean'] = gate_mean
        losses_log[f'gate_{task_name}_std'] = gate_std
        losses_log[f'lambda_{task_name}'] = gate_reg_weights.get(task_name, 1.0)
    
    tag_hits_k = {}
    if hits_k and "tag" in targets:
        tag_hits_k["hits_1"] = hits_at_k(outputs["tag"], targets["tag"], k=1)
        tag_hits_k["hits_3"] = hits_at_k(outputs["tag"], targets["tag"], k=3)
        tag_hits_k["hits_5"] = hits_at_k(outputs["tag"], targets["tag"], k=5)

    return (total_loss, losses_log, tag_hits_k) if hits_k else (total_loss, losses_log)


# ============================================================================
# Checkpoint Functions
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, epoch, step, save_path="checkpoint.pth"):
    """Save training checkpoint."""
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
    """Load training checkpoint."""
    checkpoint = torch.load(save_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    step = checkpoint['step']
    print(f"Checkpoint loaded: epoch {epoch}, step {step}")
    return epoch, step


# ============================================================================
# Configuration Management
# ============================================================================

def load_model_config(config_path):
    """
    Load model configuration saved during training.
    
    Args:
        config_path: Path to the model_config.json file
        
    Returns:
        Dictionary containing model configuration including mappings and constants
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"❌ Config file not found: {config_path}\n"
            f"   Make sure the training notebook saved model_config.json"
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    print(f"✅ Config loaded from: {config_path}")
    return config


def save_model_config(model, tag2id, time2id, scale2id, output_dir, run_name):
    """
    Save model configuration and ID mappings to JSON file.
    
    This function saves all necessary configuration for loading and testing
    the model later, ensuring consistency between training and testing.
    
    Args:
        model: MultiTaskModel instance
        tag2id: Dictionary mapping tags to indices
        time2id: Dictionary mapping time values to indices
        scale2id: Dictionary mapping scale values to indices
        output_dir: Directory to save config file
        run_name: Name for the run (used in config filename)
        
    Returns:
        str: Path to the saved config file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "run_name": run_name,
        "tag2id": tag2id,
        "time2id": time2id,
        "scale2id": scale2id,
        "id2tag": {int(v): k for k, v in tag2id.items()},
        "id2time": {int(v): k for k, v in time2id.items()},
        "id2scale": {int(v): k for k, v in scale2id.items()},
        "classification_missing_value": -100,
        "numeric_missing_value": float(torch.finfo(torch.float32).max),
        "model_args": {
            "bert_model_name": model.bert.config._name_or_path,
            "num_tags": len(tag2id),
            "num_times": len(time2id),
            "num_scales": len(scale2id),
            "hidden_size": model.bert.config.hidden_size,
        }
    }

    config_filename = f"{run_name}_config.json"
    config_path = os.path.join(output_dir, config_filename)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"✅ Configuration saved to: {config_path}")
    return config_path


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model(
    model,
    test_loader,
    device,
    task_weights,
    error_file_path,
    id2scale,
    classification_missing_value=-100,
    numeric_missing_value=1e32,
    save_errors=False,
    verbose=True,
    tag_loss_fn=None,
    time_loss_fn=None,
    scale_loss_fn=None,
    neg_loss=None
):
    """
    Evaluate the multi-task model on test data.
    
    Args:
        model: The multi-task model to evaluate.
        test_loader: DataLoader for test data.
        device: Device to run model on (cuda/cpu).
        task_weights: Dictionary of task weights for loss computation.
        error_file_path: Path prefix for saving error logs.
        id2scale: Mapping from scale ID to scale value.
        classification_missing_value: Missing value marker for classification.
        numeric_missing_value: Missing value marker for numeric data.
        save_errors: Whether to save error logs to CSV.
        verbose: Whether to show progress bar.
        
    Returns:
        Tuple of (all_losses, predictions, ground_truths, avg_hits_k, gate_df)
    """
    os.environ["WANDB_DISABLED"] = "true"
    model.eval()
    
    all_losses = {"tag": 0, "time": 0, "fact": 0, "scale": 0, "negative": 0}
    total_hits = {"hits_1": 0, "hits_3": 0, "hits_5": 0}
    predictions = defaultdict(list)
    ground_truths = defaultdict(list)
    errors = defaultdict(list)
    gate_logs = []
    
    test_batch_count = 0
    
    loader = tqdm(test_loader, desc="Evaluating") if verbose else test_loader
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
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
            loss, losses, tag_hits_k = compute_loss(outputs, targets, values, task_weights, hits_k=True, tag_loss_fn=tag_loss_fn, time_loss_fn=time_loss_fn, scale_loss_fn=scale_loss_fn, neg_loss=neg_loss)

            # Extract gate values
            if "gate" in outputs:
                gate_tensor = outputs["gate"]
            elif "gates" in outputs and isinstance(outputs["gates"], dict):
                gates_dict = outputs["gates"]
                sum_gates = sum(gates_dict.values()) 
                gate_tensor = sum_gates / len(gates_dict)
            else:
                gate_tensor = torch.zeros(input_ids.size(0), 1).to(device)

            gate_values = gate_tensor.squeeze(-1).cpu().tolist()
            batch_input_ids = input_ids.cpu().tolist()
            
            for i, (g_val, inp_id) in enumerate(zip(gate_values, batch_input_ids)):
                gate_logs.append({
                    "batch_idx": batch_idx, 
                    "sample_idx": i, 
                    "gate_value": g_val,
                    "input_ids": inp_id 
                })
            
            predictions["gate"].extend(gate_values)
            test_batch_count += 1
            
            if tag_hits_k:
                for key in total_hits:
                    total_hits[key] += tag_hits_k[key]
            
            for key in all_losses:
                if key in losses:
                    val = losses[key]
                    if isinstance(val, torch.Tensor):
                        all_losses[key] += val.item()
                    else:
                        all_losses[key] += val
            
            # Classification predictions
            for key in ["scale", "negative", "tag", "time"]:
                pred = torch.argmax(outputs[key], dim=-1).cpu().tolist()
                true = targets[key].cpu().tolist()
                
                for i, (p, t) in enumerate(zip(pred, true)):
                    predictions[key].append(p)
                    ground_truths[key].append(t)
                    if p != t:
                        errors[key].append({
                            "batch_idx": batch_idx, "sample_idx": i,
                            "true": t, "pred": p, "gate": gate_values[i]
                        })
            
            # Fact predictions (regression)
            if "fact" in targets:
                negative_pred = outputs["negative"].argmax(dim=-1).cpu().numpy()
                if "scale" in outputs:
                    scale_pred_class = outputs["scale"].argmax(dim=-1).cpu().numpy()
                else:
                    scale_pred_class = np.ones_like(values.cpu().numpy(), dtype=np.int64)
                
                id2scale_np = np.array([float(id2scale[idx]) for idx in range(len(id2scale))], dtype=np.float64)
                scale_pred_values = id2scale_np[scale_pred_class]
                
                values_np = values.cpu().numpy()
                fact_pred = values_np * ((-1) ** negative_pred) * (10 ** scale_pred_values)
                fact_target = targets["fact"].view(-1).cpu().numpy()
                
                for i, (p, t) in enumerate(zip(fact_pred, fact_target)):
                    if t == numeric_missing_value or t > 1e30: 
                        continue
                    predictions["fact"].append(p)
                    ground_truths["fact"].append(t)
                    
                    if abs(p - t) > 1e-2:
                        errors["fact"].append({
                            "batch_idx": batch_idx, "sample_idx": i, 
                            "true": float(t), "pred": float(p), "gate": gate_values[i]
                        })
    
    for key in all_losses:
        all_losses[key] /= test_batch_count if test_batch_count > 0 else 1
    
    avg_hits_k = {key: total_hits[key] / test_batch_count for key in total_hits if test_batch_count > 0}
    
    if save_errors:
        for key, error_list in errors.items():
            if error_list:
                error_file = f"{error_file_path}_errors_{key}.csv"
                fieldnames = ["batch_idx", "sample_idx", "true", "pred", "gate"]
                with open(error_file, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(error_list)

    gate_df = pd.DataFrame(gate_logs)
    
    return all_losses, dict(predictions), dict(ground_truths), avg_hits_k, gate_df


def evaluate_classification(y_true, y_pred, classification_missing_value=-100, attr=None, save_path=None):
    """
    Evaluate classification model performance.
    
    Args:
        y_true: Ground truth class labels.
        y_pred: Predicted class labels.
        classification_missing_value: Value to treat as missing (will be filtered out).
        attr: Attribute name for display.
        save_path: Optional path to save metrics to CSV file.
        
    Returns:
        dict: Dictionary containing all evaluation metrics (excluding matrices and reports).
    """
    metrics = {}
    
    # Remove missing values
    valid_indices = [i for i, t in enumerate(y_true) if t != classification_missing_value]
    y_true = [y_true[i] for i in valid_indices]
    y_pred = [y_pred[i] for i in valid_indices]

    # Calculate class statistics
    unique_classes, class_counts = np.unique(y_true, return_counts=True)
    class_weights = {cls: count / len(y_true) for cls, count in zip(unique_classes, class_counts)}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    # Weighted metrics
    weighted_precision, weighted_recall, weighted_f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    metrics['weighted_precision'] = weighted_precision
    metrics['weighted_recall'] = weighted_recall
    metrics['weighted_f1_score'] = weighted_f1_score

    # Confusion matrix and classification report (for display only, not saved to CSV)
    cm = confusion_matrix(y_true, y_pred)
    classification_rep = classification_report(y_true, y_pred, zero_division=0)
    
    # Print results
    if attr:
        print(f"\n{attr.upper()} Evaluation Metrics:")
        print(f"Weighted F1 Score: {metrics['weighted_f1_score']:.4f}")
        print(f"Weighted Recall: {metrics['weighted_recall']:.4f}")
        print(f"Weighted Precision: {metrics['weighted_precision']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score (macro): {metrics['f1_score']:.4f}")
        print(f"Precision (macro): {metrics['precision']:.4f}")
        print(f"Recall (macro): {metrics['recall']:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_rep)
    
    # Save metrics to CSV if path is provided
    if save_path:
        metrics_copy = metrics.copy()
        metrics_copy['attribute'] = attr if attr else 'unknown'
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        df = pd.DataFrame([metrics_copy])
        df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)
        print(f"Metrics saved to {save_path}")
    
    return metrics


# ============================================================================
# Model Configuration Functions
# ============================================================================

def load_model_config(config_path):
    """
    Load the model configuration saved during training.
    
    Args:
        config_path: Path to the model_config.json file
        
    Returns:
        dict: Configuration dictionary containing tag2id, time2id, scale2id mappings
              and other model metadata
              
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"❌ Config file not found: {config_path}\n"
            f"   Make sure the training notebook saved model_config.json"
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    print(f"✅ Config loaded from: {config_path}")
    return config


def save_model_config(model, tag2id, time2id, scale2id, output_dir, run_name):
    """
    Saves the model configuration and ID mappings to a JSON file.
    
    Args:
        model: The trained MultiTaskModel instance
        tag2id: Dictionary mapping tags to indices
        time2id: Dictionary mapping time values to indices
        scale2id: Dictionary mapping scale values to indices
        output_dir: Directory to save the config file
        run_name: Name of the run (for naming the config file)
        
    Returns:
        str: Path to the saved config file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        "run_name": run_name,
        "tag2id": tag2id,
        "time2id": time2id,
        "scale2id": scale2id,
        "id2tag": {int(v): k for k, v in tag2id.items()},
        "id2time": {int(v): k for k, v in time2id.items()},
        "id2scale": {int(v): k for k, v in scale2id.items()},
        "classification_missing_value": -100,
        "numeric_missing_value": float(torch.finfo(torch.float32).max),
        "model_args": {
            "bert_model_name": model.bert.config._name_or_path,
            "num_tags": len(tag2id),
            "num_times": len(time2id),
            "num_scales": len(scale2id),
            "hidden_size": model.bert.config.hidden_size,
        }
    }

    config_filename = f"{run_name}_config.json"
    config_path = os.path.join(output_dir, config_filename)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    print(f"✅ Configuration saved to: {config_path}")
    return config_path


# ============================================================================
# Multi-Task Model & Gate Head
# ============================================================================

class GateHead(nn.Module):
    """
    Multi-Task Gate Head that outputs separate noise scores for each task.
    
    Input: [CLS] token representation
    Output: Dictionary of 4 gates (one per task)
            {"tag": [0-1], "time": [0-1], "scale": [0-1], "negative": [0-1]}
            
    Each gate score represents the "noise probability":
        1 = High Noise (Ignore this sample for this task)
        0 = Clean Data (Learn from this sample for this task)
    """
    def __init__(self, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Separate output heads for each task
        self.gate_tag = nn.Linear(hidden_size, 1)
        self.gate_time = nn.Linear(hidden_size, 1)
        self.gate_scale = nn.Linear(hidden_size, 1)
        self.gate_negative = nn.Linear(hidden_size, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: [CLS] token representation, shape (batch_size, hidden_size)
            
        Returns:
            Dictionary with per-task gate scores, each shape (batch_size, 1)
        """
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)  # Tanh is standard for BERT pooler/heads
        x = self.dropout(x)
        
        # Compute gate scores for each task
        gate_tag = self.sigmoid(self.gate_tag(x))        # (batch_size, 1)
        gate_time = self.sigmoid(self.gate_time(x))
        gate_scale = self.sigmoid(self.gate_scale(x))
        gate_negative = self.sigmoid(self.gate_negative(x))
        
        return {
            "tag": gate_tag,
            "time": gate_time,
            "scale": gate_scale,
            "negative": gate_negative
        }
from transformers import AutoModel # or whatever specific AutoModel class you use

class MultiTaskModel(nn.Module):
    """
    Multi-task model with BERT encoder and task-specific heads with gating mechanism.
    
    Supports 4 tasks:
        - Tag classification
        - Time classification
        - Scale classification
        - Negative flag classification (for fact value sign)
    """
    def __init__(self, bert_model_name, num_tags, num_times, num_scales):
        super(MultiTaskModel, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name )
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
        
        # Multi-task gate head (outputs 4 separate gates)
        self.gate_head = GateHead(hidden_size)
        
    def forward(self, input_ids, attention_mask, start_tokens, end_tokens):
        """
        Args:
            input_ids: (batch_size, seq_length)
            attention_mask: (batch_size, seq_length)
            start_tokens: (batch_size,) - start indices of target span
            end_tokens: (batch_size,) - end indices of target span
            
        Returns:
            Dictionary containing:
                - "tag": (batch_size, num_tags)
                - "time": (batch_size, num_times)
                - "scale": (batch_size, num_scales)
                - "negative": (batch_size, 2)
                - "gates": Dict with per-task gates
                    - "tag": (batch_size, 1)
                    - "time": (batch_size, 1)
                    - "scale": (batch_size, 1)
                    - "negative": (batch_size, 1)
        """
        # BERT output
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_token = sequence_output[:, 0, :]
        
        # Aggregate target embeddings
        target_embeddings = [
            sequence_output[i, start_tokens[i]:end_tokens[i] + 1].mean(dim=0)
            for i in range(input_ids.size(0))
        ]  # batch_size individual tensors, each size: (hidden_size,) 
        
        target_embeddings = torch.stack(target_embeddings)  # (batch_size, hidden_size)
        
        # Task predictions
        tag_logits = self.tag_head(target_embeddings)        # (batch_size, num_tags)
        time_logits = self.time_head(target_embeddings)      # (batch_size, num_times)
        scale_logits = self.scale_head(target_embeddings)    # (batch_size, num_scales)
        negative_logits = self.negative_head(target_embeddings)  # (batch_size, 2)
        
        # Task-specific gates
        gates = self.gate_head(cls_token)  # Dict of 4 gates, each (batch_size, 1)
        
        return {
            "tag": tag_logits,
            "time": time_logits,
            "scale": scale_logits,
            "negative": negative_logits,
            "gates": gates  # Dict: {"tag": gate, "time": gate, ...}
        }


# ============================================================================
# Loss & Metrics Functions
# ============================================================================

def hits_at_k(predictions, targets, k=5):
    """
    Calculate Hits@K metric for ranking evaluation.
    
    Args:
        predictions: (batch_size, num_classes) - prediction scores
        targets: (batch_size,) or (batch_size, num_classes) - target labels
        k: Number of top predictions to consider
        
    Returns:
        float: Average Hits@K score
    """
    valid_mask = targets != -100  # Only keep valid indices
    targets = targets[valid_mask]
    predictions = predictions[valid_mask]

    top_k_preds = torch.topk(predictions, k, dim=-1).indices  # Top-K label indices

    # Ensure targets have correct dimensions
    if targets.dim() == 1:  # If targets are indices (batch_size,)
        targets = torch.nn.functional.one_hot(targets, num_classes=predictions.size(1))

    targets = targets.float()  # Ensure float tensor

    # Check if hit in top-K (batch_size, k) → (batch_size,)
    hits = torch.any(targets.gather(1, top_k_preds), dim=1).float()

    return hits.mean().item()  # Return average hit rate


def compute_loss(outputs, targets, values, task_weights=None, gate_reg_weights=None, hits_k=False,tag_loss_fn=None, time_loss_fn=None, scale_loss_fn=None, neg_loss=None):
    """
    Compute loss with task-specific gating and regularization.
    
    Formula for each task:
        L_task_final = (1 - g_task) * L_task_raw + lambda_task * (g_task)^2
    
    Total Loss = sum(task_weights[task] * L_task_final for all tasks)
    
    Args:
        outputs: Dict from model forward pass, includes "gates" key
        targets: Dict of target tensors
        values: Tensor of numeric values (for fact prediction)
        task_weights: Dict mapping task names to their importance weights
        gate_reg_weights: Dict mapping task names to gate regularization lambdas
                         Controls how strongly the gate is regularized
                         Higher lambda = harder to ignore samples for this task
                         Default is 1.0 for all tasks
        hits_k: Whether to compute Hits@K metrics for tag task
        
    Returns:
        (total_loss, losses_log) if hits_k=False
        (total_loss, losses_log, tag_hits_k) if hits_k=True
    """
    # Loss functions must be defined globally or passed as arguments
    # Using globals() to access them from the calling module
    # global tag_loss_fn, time_loss_fn, scale_loss_fn, neg_loss
    
    # Default values
    if task_weights is None:
        task_weights = {"tag": 1.0, "time": 1.0, "scale": 1.0, "negative": 1.0}
    if gate_reg_weights is None:
        gate_reg_weights = {"tag": 1.0, "time": 1.0, "scale": 1.0, "negative": 1.0}
    
    losses = {}
    
    # --- 1. Calculate Raw Vector Losses [batch_size] ---
    if "tag" in targets:
        losses["tag"] = tag_loss_fn(outputs["tag"], targets["tag"])

    if "time" in targets:
        losses["time"] = time_loss_fn(outputs["time"], targets["time"])
    
    if "scale" in targets:
        losses["scale"] = scale_loss_fn(outputs["scale"], targets["scale"])
    
    if "negative" in targets:
        losses["negative"] = neg_loss(outputs["negative"], targets["negative"])
    
    # --- 2. Extract Gates (per-task) ---
    gates = outputs.get("gates", {})  # Dict of gates, each (batch_size, 1)
    
    # Initialize total loss vector
    device = list(losses.values())[0].device
    total_loss_vec = torch.zeros(losses[list(losses.keys())[0]].shape[0], device=device)
    
    # --- 3. Apply Gate-based Loss Weighting Per Task ---
    # Formula: L_task_final = (1 - g_task) * L_task_raw + lambda_task * (g_task)^2
    for task_name, raw_loss in losses.items():
        gate = gates[task_name].squeeze(-1)  # (batch_size, 1) -> (batch_size,)
        lambda_task = gate_reg_weights.get(task_name, 1.0)
        task_weight = task_weights.get(task_name, 1.0)
        
        # Apply gate weighting
        gated_loss = (1 - gate) * raw_loss + lambda_task * gate.pow(2)
        
        # Add task-weighted contribution to total loss
        total_loss_vec += task_weight * gated_loss
    
    # --- 4. Final Average Loss ---
    total_loss = total_loss_vec.mean()
    
    # --- 5. Prepare Logging Dict (Convert vectors to scalar means for display) ---
    losses_log = {k: v.mean().item() for k, v in losses.items()}
    
    # Log gate statistics per task
    for task_name, gate in gates.items():
        gate_mean = gate.mean().item()
        gate_std = gate.std().item() if gate.numel() > 1 else 0.0
        losses_log[f'gate_{task_name}_mean'] = gate_mean
        losses_log[f'gate_{task_name}_std'] = gate_std
        losses_log[f'lambda_{task_name}'] = gate_reg_weights.get(task_name, 1.0)
    
    # --- 6. Hits@K (Optional) ---
    tag_hits_k = {}
    if hits_k and "tag" in targets:
        tag_hits_k["hits_1"] = hits_at_k(outputs["tag"], targets["tag"], k=1)
        tag_hits_k["hits_3"] = hits_at_k(outputs["tag"], targets["tag"], k=3)
        tag_hits_k["hits_5"] = hits_at_k(outputs["tag"], targets["tag"], k=5)

    return (total_loss, losses_log, tag_hits_k) if hits_k else (total_loss, losses_log)


def evaluate_model(
    model,
    test_loader,
    device,
    task_weights,
    error_file_path,
    save_errors=False,
    verbose=True,
    tag_loss_fn=None, time_loss_fn=None, scale_loss_fn=None, neg_loss=None
):
    """
    Evaluate the model on test data with comprehensive metrics.
    Modified to handle models returning a dictionary of multiple gates.
    
    Args:
        model: The trained multi-task model
        test_loader: DataLoader for test data
        device: Device to run model on (cuda/cpu)
        task_weights: Dict of task weights
        error_file_path: Base path for saving error files
        save_errors: Whether to save error samples to CSV
        verbose: Whether to print progress
        
    Returns:
        Tuple of (all_losses, predictions, ground_truths, avg_hits_k, gate_df)
    """
    os.environ["WANDB_DISABLED"] = "true"
    model.eval()
    
    all_losses = {"tag": 0, "time": 0, "fact": 0, "scale": 0, "negative": 0}
    total_hits = {"hits_1": 0, "hits_3": 0, "hits_5": 0}
    predictions = defaultdict(list)
    ground_truths = defaultdict(list)
    errors = defaultdict(list)
    
    gate_logs = []
    
    test_batch_count = 0
    fact_results = []
    
    loader = tqdm(test_loader, desc="Evaluating") if verbose else test_loader
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
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
            loss, losses, tag_hits_k = compute_loss(outputs, targets, values, task_weights, hits_k=True, tag_loss_fn=tag_loss_fn, time_loss_fn=time_loss_fn, scale_loss_fn=scale_loss_fn, neg_loss=neg_loss)

            # Extract gate values
            if "gate" in outputs:
                gate_tensor = outputs["gate"]
            elif "gates" in outputs and isinstance(outputs["gates"], dict):
                gates_dict = outputs["gates"]
                sum_gates = sum(gates_dict.values()) 
                gate_tensor = sum_gates / len(gates_dict)
            else:
                gate_tensor = torch.zeros(input_ids.size(0), 1).to(device)

            gate_values = gate_tensor.squeeze(-1).cpu().tolist()
            batch_input_ids = input_ids.cpu().tolist()
            
            for i, (g_val, inp_id) in enumerate(zip(gate_values, batch_input_ids)):
                gate_logs.append({
                    "batch_idx": batch_idx, 
                    "sample_idx": i, 
                    "gate_value": g_val,
                    "input_ids": inp_id 
                })
            
            predictions["gate"].extend(gate_values)
            test_batch_count += 1
            
            if tag_hits_k:
                for key in total_hits:
                    total_hits[key] += tag_hits_k[key]
            
            for key in all_losses:
                if key in losses:
                    val = losses[key]
                    if isinstance(val, torch.Tensor):
                        all_losses[key] += val.item()
                    else:
                        all_losses[key] += val
            
            # Classification predictions
            for key in ["scale", "negative", "tag", "time"]:
                pred = torch.argmax(outputs[key], dim=-1).cpu().tolist()
                true = targets[key].cpu().tolist()
                
                for i, (p, t) in enumerate(zip(pred, true)):
                    predictions[key].append(p)
                    ground_truths[key].append(t)
                    if p != t:
                        errors[key].append({
                            "batch_idx": batch_idx, "sample_idx": i,
                            "true": t, "pred": p, "gate": gate_values[i]
                        })
            
            # Fact predictions
            if "fact" in targets:
                negative_pred = outputs["negative"].argmax(dim=-1).cpu().numpy()
                if "scale" in outputs:
                    scale_pred_class = outputs["scale"].argmax(dim=-1).cpu().numpy()
                else:
                    scale_pred_class = np.ones_like(values.cpu().numpy(), dtype=np.int64)
                
                # Placeholder - requires id2scale to be available
                # id2scale_np would be defined in calling context
                scale_pred_values = np.ones_like(scale_pred_class, dtype=np.float64)
                
                values_np = values.cpu().numpy()
                fact_pred = values_np * ((-1) ** negative_pred) * (10 ** scale_pred_values)
                fact_target = targets["fact"].view(-1).cpu().numpy()
                
                for i, (p, t) in enumerate(zip(fact_pred, fact_target)):
                    if t > 1e30: continue
                    predictions["fact"].append(p)
                    ground_truths["fact"].append(t)
                    
                    if abs(p - t) > 1e-2:
                        errors["fact"].append({
                            "batch_idx": batch_idx, "sample_idx": i, 
                            "true": float(t), "pred": float(p), "gate": gate_values[i]
                        })
    
    for key in all_losses:
        all_losses[key] /= test_batch_count if test_batch_count > 0 else 1
    
    avg_hits_k = {key: total_hits[key] / test_batch_count for key in total_hits if test_batch_count > 0}
    
    if save_errors:
        for key, error_list in errors.items():
            if error_list:
                error_file = f"{error_file_path}_errors_{key}.csv"
                fieldnames = ["batch_idx", "sample_idx", "true", "pred", "gate"]
                with open(error_file, mode="w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(error_list)
                if verbose:
                    print(f"🚨 {len(error_list)} {key} errors saved to {error_file}")

    gate_df = pd.DataFrame(gate_logs)
    return all_losses, dict(predictions), dict(ground_truths), avg_hits_k, gate_df


def evaluate_classification(y_true, y_pred, attr=None, plot_confusion=False, save_path=None):
    """
    Evaluate classification model performance with comprehensive metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        attr: Attribute name for display
        plot_confusion: Whether to plot confusion matrix
        save_path: Optional path to save metrics to CSV file
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    y_true = np.array(y_true, dtype=np.int32)
    y_pred = np.array(y_pred, dtype=np.int32)
    
    # Filter out -100 values (ignore index)
    mask = y_true != -100
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Calculate metrics
    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    precision = precision_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    recall = recall_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    f1 = f1_score(y_true_filtered, y_pred_filtered, average='weighted', zero_division=0)
    
    macro_precision = precision_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    macro_recall = recall_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    macro_f1 = f1_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    # Print results
    if attr:
        print(f"\n{attr.upper()} Classification Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")


        print(f"Precision (macro): {macro_precision:.4f}")
        print(f"Recall (macro): {macro_recall:.4f}")
        print(f"F1 Score (macro): {macro_f1:.4f}")
        
        # Confusion matrix and report
        cm = confusion_matrix(y_true_filtered, y_pred_filtered)
        classification_rep = classification_report(y_true_filtered, y_pred_filtered, zero_division=0)
        print(f"Confusion Matrix:")
        print(cm)
        print(f"Classification Report:")
        print(classification_rep)
    
    # Plot confusion matrix if requested
    if plot_confusion:
        cm = confusion_matrix(y_true_filtered, y_pred_filtered)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f"{attr.upper()} Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(np.unique(y_true_filtered)))
        plt.xticks(tick_marks, np.unique(y_true_filtered))
        plt.yticks(tick_marks, np.unique(y_true_filtered))
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.show()
    
    # Save metrics to CSV if path is provided
    if save_path:
        metrics_copy = metrics.copy()
        metrics_copy['attribute'] = attr if attr else 'unknown'
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        df = pd.DataFrame([metrics_copy])
        df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)
        print(f"Metrics saved to {save_path}")
    
    return metrics


# ============================================================================
# Gate-Based Filtering & Analysis
# ============================================================================

def evaluate_model_with_gate_filter(model, test_loader, device, id2scale, gate_threshold=0.5, tag_loss_fn=None, time_loss_fn=None,):
    """
    Evaluates the model on samples where the average gate value <= threshold.
    Handles both single-gate and multi-gate models.
    
    Args:
        model: The multi-task model to evaluate.
        test_loader: DataLoader for test data.
        device: Device to run model on (cuda/cpu).
        id2scale: Mapping from scale ID to scale value.
        gate_threshold: Threshold for filtering samples (lower = cleaner data).
        
    Returns:
        Tuple of (predictions, ground_truths, coverage)
    """
    model.to(device) 
    model.eval()
    
    predictions = defaultdict(list)
    ground_truths = defaultdict(list)
    
    total_samples = 0
    kept_samples = 0
    
    # Prepare id2scale as numpy array for efficiency
    id2scale_np = np.array([float(id2scale[idx]) for idx in range(len(id2scale))], dtype=np.float64)

    with torch.no_grad():
        for batch in test_loader:
            # Move inputs to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_tokens = batch["start_token"].to(device)
            end_tokens = batch["end_token"].to(device)
            values = batch["value"].to(device)
            
            # Move targets to device
            targets = {
                "tag": batch["tag"].to(device),
                "time": batch["time"].to(device),
                "fact": batch["fact"].to(device),
                "scale": batch["scale"].to(device),
                "negative": batch["negative"].to(device)
            }
            
            # Forward Pass
            outputs = model(input_ids, attention_mask, start_tokens, end_tokens)
            
            # --- GATE HANDLING LOGIC (Supports both single and multi-gate models) ---
            if "gate" in outputs:
                # Case 1: Single gate tensor
                gate_tensor = outputs["gate"]
            elif "gates" in outputs and isinstance(outputs["gates"], dict):
                # Case 2: Dict of multiple gates -> Average them for global noise score
                gates_dict = outputs["gates"]
                sum_gates = sum(gates_dict.values())
                gate_tensor = sum_gates / len(gates_dict)
            else:
                # Fallback
                gate_tensor = torch.zeros(input_ids.size(0), 1).to(device)

            # Flatten to [batch_size] array
            gate_values = gate_tensor.squeeze(-1).cpu().numpy()
            
            # Get predictions
            batch_preds = {}
            for key in ["tag", "time", "scale", "negative"]:
                batch_preds[key] = torch.argmax(outputs[key], dim=-1).cpu().numpy()
            
            # Process Fact Prediction
            neg_pred = batch_preds["negative"]
            scale_idx = batch_preds["scale"]
            scale_val = id2scale_np[scale_idx]
            vals = values.cpu().numpy()
            
            fact_preds = vals * ((-1) ** neg_pred) * (10 ** scale_val)

            # Move targets to CPU
            batch_targets = {k: v.cpu().numpy() for k, v in targets.items()}

            # Filter samples based on gate threshold
            batch_size = input_ids.size(0)
            for i in range(batch_size):
                total_samples += 1
                
                # Filter Logic: Keep if gate value (noise score) is low
                if gate_values[i] > gate_threshold:
                    continue 
                
                kept_samples += 1
                
                # Store Classification predictions
                for key in ["tag", "time", "scale", "negative"]:
                    t = batch_targets[key][i]
                    if t != -100:  # Skip missing values
                        predictions[key].append(batch_preds[key][i])
                        ground_truths[key].append(t)
                
                # Store Fact Regression prediction
                if "fact" in batch_targets:
                    t_fact = batch_targets["fact"][i]
                    if t_fact < 1e30:  # Check for missing value
                        predictions["fact"].append(fact_preds[i])
                        ground_truths["fact"].append(t_fact)

    coverage = kept_samples / total_samples if total_samples > 0 else 0
    return dict(predictions), dict(ground_truths), coverage


def evaluate_regression(y_true, y_pred, attr=None, plot_residuals=False, save_path=None):
    """
    Evaluate regression model performance.
    
    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        attr: Attribute name for display.
        plot_residuals: Whether to plot residual plot.
        save_path: Optional path to save metrics to CSV file.
        
    Returns:
        dict: Dictionary containing all evaluation metrics.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    metrics = {}
    
    # Calculate regression metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # Print results
    if attr:
        print(f"\n{attr.upper()} Regression Metrics:")
        print(f"MSE: {metrics['mse']:.4f}")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R² Score: {metrics['r2']:.4f}")
    
    # Plot residuals if requested
    if plot_residuals:
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 6))
        plt.scatter(y_pred, residuals, alpha=0.6, color='blue', edgecolors='black')
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel("Predicted Values")
        plt.ylabel("Residuals")
        plt.title(f"{attr.upper()} Residual Plot")
        plt.grid()
        plt.show()
    
    # Save metrics to CSV if path is provided
    if save_path:
        metrics_copy = metrics.copy()
        metrics_copy['attribute'] = attr if attr else 'unknown'
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        df = pd.DataFrame([metrics_copy])
        df.to_csv(save_path, mode='a', header=not os.path.exists(save_path), index=False)
        print(f"Metrics saved to {save_path}")
    
    return metrics
