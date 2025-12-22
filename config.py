"""
Configuration Template for SecBERT Training

This file provides default configuration values and helps users
understand the key hyperparameters for training the model.

To use this:
1. Copy this file to your project: cp config_template.py config.py
2. Modify values as needed for your dataset
3. Import in your training script: from config import CONFIG
"""

# ============================================
# Data Paths
# ============================================

DATA_CONFIG = {
    # Training data (JSONL files with raw jobs)
    "train_files": [
        "processed_data_task1_smaller/train_400k.jsonl",
    ],
    
    # Validation data
    "valid_files": [
        "processed_data_task1_smaller/valid_50k.jsonl",
    ],
    
    # Test data
    "test_files": [
        "processed_data_task1_smaller/test_50k.jsonl",
    ],
    
    # Pre-computed label count files (for class weights)
    "label_counts_dir": "processed_iterable_dataset/counter/",
}


# ============================================
# Model Architecture
# ============================================

MODEL_CONFIG = {
    # Base BERT model
    "bert_model_name": "nlpaueb/sec-bert-base",
    
    # Task dimensions
    # These should match your data!
    # Note: You must run data analysis to determine exact numbers
    "num_tags": 200,              # Number of unique tags
    "num_times": 9,               # Time classifications
    "num_scales": 25,             # Scale values from -12 to +12
    
    # Hidden layer sizes for task heads
    "hidden_size": 768,           # BERT hidden size (usually 768)
    "task_head_reduction": 2,     # Reduce to hidden_size // 2
    
    # Dropout rates
    "dropout_rates": {
        "gate_head": 0.1,
        "tag_head": 0.3,
        "time_head": 0.3,
        "scale_head": 0.5,
        "negative_head": 0.5,
    },
    
    # Tokenization
    "max_length": 512,
    "tokenizer_name": "nlpaueb/sec-bert-base",
}


# ============================================
# Training Hyperparameters
# ============================================

TRAINING_CONFIG = {
    # Basic settings
    "num_epochs": 30,
    "batch_size": 32,
    "num_workers": 4,
    "seed": 22,
    
    # Early stopping
    "patience": 3,
    "eval_every": 10000,          # Steps between validations
    "log_every": 100,             # Steps between logging
    
    # Gradient control
    "max_grad_norm": 1.0,         # Gradient clipping
    "warmup_steps": 5,            # Linear warmup steps
    
    # Optimizer settings
    "optimizer": "AdamW",
    "weight_decay": 1e-2,
}


# ============================================
# Learning Rates
# ============================================

LEARNING_RATES = {
    # BERT gets lower LR (fine-tuning)
    "bert": 1e-5,
    
    # Task heads get higher LRs (random initialization)
    "tag_head": 5e-4,
    "time_head": 1e-5,
    "scale_head": 3e-5,
    "negative_head": 2e-5,
    
    # Gate head uses same LR as BERT
    "gate_head": 1e-5,
}


# ============================================
# Loss Function Configuration
# ============================================

LOSS_CONFIG = {
    # Task weights: balance importance across tasks
    # Higher = more weight in total loss
    "task_weights": {
        "tag": 10.0,      # Tag is primary task
        "time": 1.0,      # Equal weight
        "scale": 0.2,     # Easier task, lower weight
        "negative": 5.0,  # Important for numeric interpretation
    },
    
    # Gate regularization weights (lambda in the formula)
    # Higher = more resistant to gating (harder to ignore)
    "gate_reg_weights": {
        "tag": 1.0,
        "time": 1.0,
        "scale": 1.0,
        "negative": 1.0,
    },
    
    # Class-balanced loss settings
    "cb_loss_beta": 0.99,  # Hyperparameter for CB_CE_Loss
    
    # Focal loss settings
    "focal_loss": {
        "alpha": 0.25,     # Weighting factor
        "gamma": 3.0,      # Focusing parameter
    },
    
    # Missing value sentinels
    "classification_missing": -100,
    "numeric_missing": float('3.4028235e+38'),  # torch.finfo(torch.float32).max
}


# ============================================
# Scheduler Configuration
# ============================================

SCHEDULER_CONFIG = {
    "scheduler_type": "CosineAnnealingLR",
    
    # CosineAnnealingLR
    "cosine_annealing": {
        "T_max": "num_total_steps // 4",  # Evaluated dynamically
        "eta_min": 1e-7,                  # Minimum learning rate
    },
}


# ============================================
# Output Paths
# ============================================

PATHS_CONFIG = {
    # Model weights
    "model_dir": "model_weight/secbert/",
    
    # Training checkpoints
    "checkpoint_dir": "check_point/secbert/",
    
    # Evaluation results
    "result_dir": "result/secbert/",
    
    # Logs
    "log_dir": "logs/",
}


# ============================================
# Weights & Biases Configuration
# ============================================

WANDB_CONFIG = {
    "enabled": True,
    "project": "multi-task-model",
    "entity": None,  # Set to your W&B team name if applicable
    "log_frequency": 100,
    "watch_model": True,
}


# ============================================
# Data Processing
# ============================================

DATA_PROCESSING_CONFIG = {
    # Tokenizer settings
    "tokenizer": {
        "padding": "max_length",
        "truncation": True,
        "max_length": 512,
        "return_offsets_mapping": True,
        "return_special_tokens_mask": True,
    },
    
    # Batch processing
    "batch_processing": {
        "batch_size": 32,
        "num_workers": 8,  # ThreadPoolExecutor workers
    },
    
    # Dataset buffering
    "buffered_shuffle": {
        "buffer_size": 8000,  # Number of samples to buffer
    },
    
    # Rare tag handling
    "rare_tag_threshold": 10,  # Tags with count < this are rare
    "rare_tag_mapping": "standard_rare",
}


# ============================================
# Label Configurations
# ============================================

LABEL_CONFIG = {
    # These are populated from your data
    # Example structures shown below
    
    # Time dimension values (should match your data)
    "time_classes": [
        "instant; past",
        "instant; current",
        "instant; future",
        "period; past",
        "period; current",
        "period; future",
        "period; past_current",
        "period; current_future",
        "period; past_future",
    ],
    
    # Scale dimension (powers of 10)
    "scale_classes": [str(i) for i in range(-12, 13)],
    
    # Negative dimension (binary)
    "negative_classes": [0, 1],  # 0: positive, 1: negative
    
    # Tag classes must be loaded from your data
    # Too many to list here
    "tag_classes_file": "processed_iterable_dataset/counter/tag_count_train_400k.json",
}


# ============================================
# Validation Configuration
# ============================================

VALIDATION_CONFIG = {
    # Metrics to compute
    "metrics": {
        "classification": ["precision", "recall", "f1", "confusion_matrix"],
        "ranking": ["hits@1", "hits@3", "hits@5"],
        "gates": ["mean", "std", "min", "max", "median"],
    },
    
    # Error analysis
    "save_errors": True,
    "max_errors_per_task": 1000,  # Limit error CSVs to N samples
}


# ============================================
# Advanced Settings
# ============================================

ADVANCED_CONFIG = {
    # Mixed precision training (experimental)
    "use_amp": False,  # Set to True to enable
    
    # Gradient accumulation
    "accumulation_steps": 1,
    
    # Distributed training
    "distributed": False,
    
    # Device settings
    "device": "cuda",  # or "cpu"
    "pin_memory": True,
    "num_workers_dataloader": 4,
    
    # Profiling
    "profile": False,
}


# ============================================
# Complete Configuration Object
# ============================================

CONFIG = {
    "data": DATA_CONFIG,
    "model": MODEL_CONFIG,
    "training": TRAINING_CONFIG,
    "learning_rates": LEARNING_RATES,
    "loss": LOSS_CONFIG,
    "scheduler": SCHEDULER_CONFIG,
    "paths": PATHS_CONFIG,
    "wandb": WANDB_CONFIG,
    "data_processing": DATA_PROCESSING_CONFIG,
    "labels": LABEL_CONFIG,
    "validation": VALIDATION_CONFIG,
    "advanced": ADVANCED_CONFIG,
}


# ============================================
# Usage Example
# ============================================

if __name__ == "__main__":
    """
    Example of how to use this configuration.
    """
    
    print("SecBERT Configuration Template")
    print("="*60)
    
    print("\nModel Configuration:")
    print(f"  BERT: {CONFIG['model']['bert_model_name']}")
    print(f"  Tags: {CONFIG['model']['num_tags']}")
    print(f"  Times: {CONFIG['model']['num_times']}")
    print(f"  Scales: {CONFIG['model']['num_scales']}")
    
    print("\nTraining Configuration:")
    print(f"  Epochs: {CONFIG['training']['num_epochs']}")
    print(f"  Batch Size: {CONFIG['training']['batch_size']}")
    print(f"  Learning Rates: {CONFIG['learning_rates']}")
    
    print("\nLoss Configuration:")
    print(f"  Task Weights: {CONFIG['loss']['task_weights']}")
    print(f"  Gate Regularization: {CONFIG['loss']['gate_reg_weights']}")
    
    print("\nData Configuration:")
    print(f"  Training Files: {CONFIG['data']['train_files']}")
    print(f"  Validation Files: {CONFIG['data']['valid_files']}")
    
    print("\nPaths:")
    for key, path in CONFIG['paths'].items():
        print(f"  {key}: {path}")
    
    print("\n" + "="*60)
    print("To customize: Copy this file to 'config.py' and edit values")
