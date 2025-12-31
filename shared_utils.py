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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from word2number import w2n
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import BertModel


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
