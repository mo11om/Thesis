"""
Example Training Script for SecBERT Multi-Task Model

This script demonstrates how to:
1. Load configuration
2. Initialize model and data loaders
3. Set up optimizer and scheduler
4. Run training loop with validation
5. Save checkpoints and model config

Usage:
    python train_example.py --config-dir model_weight/secbert \
        --run-name secbert-0426-173 --batch-size 32 --epochs 30
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

# Import from the module
import sys
sys.path.insert(0, os.path.dirname(__file__))

from model import MultiTaskModel
from preprocess import MultiTaskIterableDataset, BufferedShuffleDataset
from train import (
    set_seed, CB_CE_Loss, FocalLoss, compute_loss,
    save_checkpoint, load_checkpoint, save_model_config,
    validate_model
)
import wandb


def create_data_loaders(train_files, valid_files, batch_size, tokenizer, 
                       tag2id, time2id, scale2id, device, num_workers=4):
    """
    Create train and validation data loaders.
    
    Args:
        train_files (list): Training JSONL file paths
        valid_files (list): Validation JSONL file paths
        batch_size (int): Batch size
        tokenizer: HuggingFace tokenizer
        tag2id, time2id, scale2id (dict): Label mappings
        device (str): Device to use
        num_workers (int): Number of data loading workers
        
    Returns:
        tuple: (train_loader, valid_loader)
    """
    target_attrs = ["tag", "time", "scale", "negative", "fact"]
    
    # Create raw datasets
    train_dataset = MultiTaskIterableDataset(
        files=train_files,
        target_attrs=target_attrs,
        tokenizer=tokenizer,
        tag2id=tag2id,
        time2id=time2id,
        scale2id=scale2id,
        batch_size=32,
        num_workers=num_workers
    )
    
    valid_dataset = MultiTaskIterableDataset(
        files=valid_files,
        target_attrs=target_attrs,
        tokenizer=tokenizer,
        tag2id=tag2id,
        time2id=time2id,
        scale2id=scale2id,
        batch_size=32,
        num_workers=num_workers
    )
    
    # Add shuffling to training set
    train_dataset = BufferedShuffleDataset(train_dataset, buffer_size=8000)
    
    # Create data loaders
    def collate_fn(batch):
        """Custom collate function to handle tensor stacking."""
        result = {}
        for key in batch[0].keys():
            if key in ["input_ids", "attention_mask", "start_token", "end_token", "value",
                      "tag", "time", "scale", "negative", "fact"]:
                result[key] = torch.stack([torch.tensor(item[key]) if not isinstance(item[key], torch.Tensor) 
                                          else item[key] for item in batch])
            else:
                result[key] = [item[key] for item in batch]
        return result
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,  # IterableDataset doesn't use num_workers in main process
        pin_memory=True if device == "cuda" else False
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if device == "cuda" else False
    )
    
    return train_loader, valid_loader


def main(args):
    """Main training function."""
    
    # ===== Setup =====
    set_seed(22)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # ===== Load Tokenizer =====
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-base")
    
    # ===== Load/Build Label Mappings =====
    # In production, these would come from a data analysis step
    # For now, we load from a config if available, or use defaults
    
    tag_list = []  # Load from your data
    time_list = ['instant; past', 'instant; current', 'instant; future',
                 'period; past', 'period; current', 'period; future',
                 'period; past_current', 'period; current_future', 'period; past_future']
    scale_list = [str(i) for i in range(-12, 13)]
    
    tag2id = {tag: idx for idx, tag in enumerate(tag_list)}
    time2id = {time: idx for idx, time in enumerate(time_list)}
    scale2id = {scale: idx for idx, scale in enumerate(scale_list)}
    
    print(f"Loaded {len(tag2id)} tags, {len(time2id)} times, {len(scale2id)} scales")
    
    # ===== Initialize Model =====
    print("Initializing model...")
    model = MultiTaskModel(
        bert_model_name="nlpaueb/sec-bert-base",
        num_tags=len(tag2id),
        num_times=len(time2id),
        num_scales=len(scale2id)
    )
    model.to(device)
    
    # ===== Create Loss Functions =====
    # Note: In full implementation, these would use actual class weights from training data
    num_tag_samples = [100] * len(tag2id)  # Placeholder
    
    tag_loss_fn = CB_CE_Loss(num_tag_samples, beta=0.99, ignore_index=-100)
    time_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    scale_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    neg_loss = FocalLoss(alpha=0.25, gamma=3.0, reduction="none")
    
    # ===== Setup Optimizer =====
    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": args.bert_lr, "weight_decay": 1e-2},
        {"params": model.tag_head.parameters(), "lr": args.tag_head_lr, "weight_decay": 1e-2},
        {"params": model.time_head.parameters(), "lr": args.time_head_lr, "weight_decay": 1e-2},
        {"params": model.scale_head.parameters(), "lr": args.scale_head_lr, "weight_decay": 1e-2},
        {"params": model.negative_head.parameters(), "lr": args.negative_head_lr, "weight_decay": 1e-2},
    ])
    
    # ===== Setup Scheduler =====
    num_train_steps = args.epochs * (1000000 // args.batch_size)  # Approximate
    scheduler = CosineAnnealingLR(optimizer, T_max=num_train_steps // 4, eta_min=1e-7)
    
    # ===== Setup W&B Logging =====
    if not args.no_wandb:
        wandb.init(
            project="multi-task-model",
            name=args.run_name,
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rates": {
                    "bert": args.bert_lr,
                    "tag_head": args.tag_head_lr,
                    "time_head": args.time_head_lr,
                    "scale_head": args.scale_head_lr,
                    "negative_head": args.negative_head_lr
                },
                "model_architecture": "SecBERT + GateHead"
            }
        )
    
    # ===== Load Data =====
    print("Loading data...")
    train_loader, valid_loader = create_data_loaders(
        train_files=args.train_files.split(','),
        valid_files=args.valid_files.split(','),
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        tag2id=tag2id,
        time2id=time2id,
        scale2id=scale2id,
        device=device,
        num_workers=args.num_workers
    )
    
    # ===== Training Loop =====
    task_weights = {"tag": 1.0, "time": 1.0, "scale": 1.0, "negative": 1.0}
    gate_reg_weights = {"tag": 1.0, "time": 1.0, "scale": 1.0, "negative": 1.0}
    
    best_val_loss = float("inf")
    early_stop_counter = 0
    global_step = 0
    
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        if early_stop_counter >= args.patience:
            print(f"Early stopping triggered after {early_stop_counter} epochs without improvement")
            break
        
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}") as pbar:
            for batch in pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_tokens = batch["start_token"].to(device)
                end_tokens = batch["end_token"].to(device)
                values = batch.get("value", torch.zeros(len(input_ids))).to(device)
                
                targets = {
                    "tag": batch["tag"].to(device),
                    "time": batch["time"].to(device),
                    "scale": batch["scale"].to(device),
                    "negative": batch["negative"].to(device)
                }
                
                # Forward pass
                outputs = model(input_ids, attention_mask, start_tokens, end_tokens)
                loss, losses = compute_loss(outputs, targets, values, task_weights, gate_reg_weights)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                pbar.set_postfix({"loss": loss.item()})
                
                # Log to W&B
                if not args.no_wandb and global_step % args.log_steps == 0:
                    wandb.log({
                        "step": global_step,
                        "epoch": epoch,
                        "train_loss": loss.item(),
                        **{f"train_{k}": v for k, v in losses.items()}
                    })
                
                # Validation
                if global_step % args.eval_steps == 0:
                    val_loss, val_losses = validate_model(
                        model, valid_loader, task_weights, gate_reg_weights, device
                    )
                    
                    print(f"\nStep {global_step}: Val Loss = {val_loss:.4f}")
                    
                    if not args.no_wandb:
                        wandb.log({
                            "step": global_step,
                            "val_loss": val_loss,
                            **{f"val_{k}": v for k, v in val_losses.items()}
                        })
                    
                    # Save best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        early_stop_counter = 0
                        
                        model_path = os.path.join(
                            args.model_dir,
                            f"{args.run_name}_step{global_step}.pt"
                        )
                        torch.save(model.state_dict(), model_path)
                        print(f"✅ Saved best model to {model_path}")
                    else:
                        early_stop_counter += 1
        
        print(f"Epoch {epoch+1} average loss: {epoch_loss / num_batches:.4f}")
    
    # ===== Save Final Model Config =====
    print("\nSaving model configuration...")
    save_model_config(
        model=model,
        tag2id=tag2id,
        time2id=time2id,
        scale2id=scale2id,
        num_tags=len(tag2id),
        num_times=len(time2id),
        num_scales=len(scale2id),
        output_dir=args.model_dir,
        run_name=args.run_name
    )
    
    if not args.no_wandb:
        wandb.finish()
    
    print("✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SecBERT Multi-Task Model")
    
    # Data paths
    parser.add_argument("--train-files", type=str, default="processed_data_task1_smaller/train_400k.jsonl",
                       help="Comma-separated list of training JSONL files")
    parser.add_argument("--valid-files", type=str, default="processed_data_task1_smaller/valid_50k.jsonl",
                       help="Comma-separated list of validation JSONL files")
    
    # Output paths
    parser.add_argument("--model-dir", type=str, default="model_weight/secbert",
                       help="Directory to save model weights")
    parser.add_argument("--checkpoint-dir", type=str, default="check_point/secbert",
                       help="Directory to save checkpoints")
    
    # Model config
    parser.add_argument("--run-name", type=str, default="secbert-0426-173",
                       help="Name of the training run")
    
    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--eval-steps", type=int, default=10000)
    parser.add_argument("--log-steps", type=int, default=100)
    
    # Learning rates
    parser.add_argument("--bert-lr", type=float, default=1e-5)
    parser.add_argument("--tag-head-lr", type=float, default=5e-4)
    parser.add_argument("--time-head-lr", type=float, default=1e-5)
    parser.add_argument("--scale-head-lr", type=float, default=3e-5)
    parser.add_argument("--negative-head-lr", type=float, default=2e-5)
    
    # Other
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    
    args = parser.parse_args()
    main(args)
