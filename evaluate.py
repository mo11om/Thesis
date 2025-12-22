"""
Model Evaluation and Inference
Provides comprehensive evaluation metrics and inference utilities

This module includes:
- Metric calculations: Hits@K, Precision, Recall, F1, MSE, RMSE, MAE, R2
- Main evaluation loop with error analysis
- Gate value extraction and interpretation
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm


def hits_at_k(predictions, targets, k=5):
    """
    Calculate Hits@K metric.
    
    Args:
        predictions (torch.Tensor): Logits of shape (batch_size, num_classes)
        targets (torch.Tensor): Target indices of shape (batch_size,)
        k (int): Top-K to evaluate
        
    Returns:
        float: Proportion of correct predictions in top-K
    """
    valid_mask = targets != -100
    if not valid_mask.any():
        return 0.0
    
    targets = targets[valid_mask]
    predictions = predictions[valid_mask]
    
    top_k_preds = torch.topk(predictions, k, dim=-1).indices
    
    if targets.dim() == 1:
        targets = F.one_hot(targets, num_classes=predictions.size(1))
    
    targets = targets.float()
    hits = torch.any(targets.gather(1, top_k_preds), dim=1).float()
    
    return hits.mean().item()


def evaluate_classification(predictions, targets, class_names=None, id2label=None):
    """
    Evaluate classification performance with standard metrics.
    
    Computes:
    - Per-class and macro/micro average Precision, Recall, F1
    - Confusion Matrix
    - Per-class statistics
    
    Args:
        predictions (np.ndarray): Predicted class indices, shape (N,)
        targets (np.ndarray): True class indices, shape (N,)
        class_names (list): Optional list of class names
        id2label (dict): Optional mapping from index to label
        
    Returns:
        dict: Comprehensive evaluation metrics
    """
    # Remove invalid targets
    valid_mask = targets != -100
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    if len(targets) == 0:
        return {
            "error": "No valid targets",
            "precision": 0, "recall": 0, "f1": 0,
            "confusion_matrix": None
        }
    
    # Standard metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, predictions,
        average=None,
        zero_division=0
    )
    
    # Macro and micro averages
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1.mean()
    
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        targets, predictions,
        average='micro',
        zero_division=0
    )
    
    # Confusion matrix
    num_classes = max(max(predictions), max(targets)) + 1
    conf_matrix = confusion_matrix(
        targets, predictions,
        labels=list(range(num_classes))
    )
    
    # Build results
    results = {
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
        "micro_f1": float(micro_f1),
        "confusion_matrix": conf_matrix.tolist(),
        "per_class": {}
    }
    
    # Per-class metrics
    for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
        label = id2label.get(i, str(i)) if id2label else str(i)
        results["per_class"][label] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "support": int(s)
        }
    
    return results


def evaluate_regression(predictions, targets, metric_name=""):
    """
    Evaluate regression performance with standard metrics.
    
    Computes:
    - MSE, RMSE: Mean and root mean squared error
    - MAE: Mean absolute error
    - R²: Coefficient of determination
    
    Args:
        predictions (np.ndarray): Predicted values, shape (N,)
        targets (np.ndarray): True values, shape (N,)
        metric_name (str): Optional name for the metric (for logging)
        
    Returns:
        dict: Regression metrics
    """
    # Remove invalid targets
    valid_mask = targets != float(torch.finfo(torch.float32).max)
    predictions = predictions[valid_mask]
    targets = targets[valid_mask]
    
    if len(targets) == 0:
        return {
            "error": "No valid targets",
            "mse": 0, "rmse": 0, "mae": 0, "r2": 0
        }
    
    # Calculate metrics
    mse = ((predictions - targets) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(predictions - targets).mean()
    
    # R² Score
    ss_res = ((targets - predictions) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "samples": len(targets)
    }


def evaluate_model(model, test_loader, device, id2tag, id2time, id2scale, output_dir,
                   run_name, classification_missing_value=-100, numeric_missing_value=None):
    """
    Full evaluation loop with error analysis and predictions saving.
    
    For each sample in the test set:
    1. Get model predictions
    2. Extract gate values (task-specific importance scores)
    3. Compute metrics per task
    4. Save detailed predictions and error analysis
    
    Args:
        model (nn.Module): Trained model to evaluate
        test_loader (DataLoader): Test data loader
        device (str): Device to run on
        id2tag (dict): Tag index to name mapping
        id2time (dict): Time index to name mapping
        id2scale (dict): Scale index to name mapping
        output_dir (str): Directory to save results
        run_name (str): Run identifier for filenames
        classification_missing_value (int): Missing value sentinel for classification
        numeric_missing_value (float): Missing value sentinel for regression
        
    Returns:
        dict: Comprehensive evaluation results
    """
    if numeric_missing_value is None:
        numeric_missing_value = float(torch.finfo(torch.float32).max)
    
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for predictions and analysis
    all_predictions = {
        "tag": {"pred": [], "true": [], "gate": []},
        "time": {"pred": [], "true": [], "gate": []},
        "scale": {"pred": [], "true": [], "gate": []},
        "negative": {"pred": [], "true": [], "gate": []},
    }
    
    # Error tracking by category
    errors_by_category = {
        "tag": [],
        "time": [],
        "scale": [],
        "negative": [],
        "fact": []
    }
    
    results = {}
    
    print(f"Starting evaluation on {len(test_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_tokens = batch["start_token"].to(device)
            end_tokens = batch["end_token"].to(device)
            
            # Get model outputs
            outputs = model(input_ids, attention_mask, start_tokens, end_tokens)
            gates = outputs["gates"]
            
            # ===== TAG EVALUATION =====
            if "tag" in batch:
                tag_pred = outputs["tag"].argmax(dim=-1).cpu().numpy()
                tag_true = batch["tag"].cpu().numpy()
                tag_gate = gates["tag"].squeeze(-1).cpu().numpy()
                
                all_predictions["tag"]["pred"].extend(tag_pred)
                all_predictions["tag"]["true"].extend(tag_true)
                all_predictions["tag"]["gate"].extend(tag_gate)
                
                # Error analysis: find misclassifications
                errors = tag_true != tag_pred
                for i, is_error in enumerate(errors):
                    if is_error and tag_true[i] != classification_missing_value:
                        errors_by_category["tag"].append({
                            "true": id2tag.get(int(tag_true[i]), str(tag_true[i])),
                            "pred": id2tag.get(int(tag_pred[i]), str(tag_pred[i])),
                            "gate": float(tag_gate[i]),
                            "batch": batch_idx,
                            "sample": i
                        })
            
            # ===== TIME EVALUATION =====
            if "time" in batch:
                time_pred = outputs["time"].argmax(dim=-1).cpu().numpy()
                time_true = batch["time"].cpu().numpy()
                time_gate = gates["time"].squeeze(-1).cpu().numpy()
                
                all_predictions["time"]["pred"].extend(time_pred)
                all_predictions["time"]["true"].extend(time_true)
                all_predictions["time"]["gate"].extend(time_gate)
                
                errors = time_true != time_pred
                for i, is_error in enumerate(errors):
                    if is_error and time_true[i] != classification_missing_value:
                        errors_by_category["time"].append({
                            "true": id2time.get(int(time_true[i]), str(time_true[i])),
                            "pred": id2time.get(int(time_pred[i]), str(time_pred[i])),
                            "gate": float(time_gate[i]),
                            "batch": batch_idx,
                            "sample": i
                        })
            
            # ===== SCALE EVALUATION =====
            if "scale" in batch:
                scale_pred = outputs["scale"].argmax(dim=-1).cpu().numpy()
                scale_true = batch["scale"].cpu().numpy()
                scale_gate = gates["scale"].squeeze(-1).cpu().numpy()
                
                all_predictions["scale"]["pred"].extend(scale_pred)
                all_predictions["scale"]["true"].extend(scale_true)
                all_predictions["scale"]["gate"].extend(scale_gate)
                
                errors = scale_true != scale_pred
                for i, is_error in enumerate(errors):
                    if is_error and scale_true[i] != classification_missing_value:
                        errors_by_category["scale"].append({
                            "true": id2scale.get(int(scale_true[i]), str(scale_true[i])),
                            "pred": id2scale.get(int(scale_pred[i]), str(scale_pred[i])),
                            "gate": float(scale_gate[i]),
                            "batch": batch_idx,
                            "sample": i
                        })
            
            # ===== NEGATIVE EVALUATION =====
            if "negative" in batch:
                neg_pred = outputs["negative"].argmax(dim=-1).cpu().numpy()
                neg_true = batch["negative"].cpu().numpy()
                neg_gate = gates["negative"].squeeze(-1).cpu().numpy()
                
                all_predictions["negative"]["pred"].extend(neg_pred)
                all_predictions["negative"]["true"].extend(neg_true)
                all_predictions["negative"]["gate"].extend(neg_gate)
                
                errors = neg_true != neg_pred
                for i, is_error in enumerate(errors):
                    if is_error and neg_true[i] != classification_missing_value:
                        errors_by_category["negative"].append({
                            "true": str(neg_true[i]),
                            "pred": str(neg_pred[i]),
                            "gate": float(neg_gate[i]),
                            "batch": batch_idx,
                            "sample": i
                        })
    
    # ===== Compute Final Metrics =====
    print("Computing metrics...")
    
    results["tag"] = evaluate_classification(
        np.array(all_predictions["tag"]["pred"]),
        np.array(all_predictions["tag"]["true"]),
        id2label=id2tag
    )
    results["tag"]["hits@3"] = hits_at_k(
        torch.tensor(np.array(all_predictions["tag"]["pred"])),
        torch.tensor(np.array(all_predictions["tag"]["true"])),
        k=3
    )
    results["tag"]["hits@5"] = hits_at_k(
        torch.tensor(np.array(all_predictions["tag"]["pred"])),
        torch.tensor(np.array(all_predictions["tag"]["true"])),
        k=5
    )
    
    results["time"] = evaluate_classification(
        np.array(all_predictions["time"]["pred"]),
        np.array(all_predictions["time"]["true"]),
        id2label=id2time
    )
    
    results["scale"] = evaluate_classification(
        np.array(all_predictions["scale"]["pred"]),
        np.array(all_predictions["scale"]["true"]),
        id2label=id2scale
    )
    
    results["negative"] = evaluate_classification(
        np.array(all_predictions["negative"]["pred"]),
        np.array(all_predictions["negative"]["true"]),
        id2label={0: "positive", 1: "negative"}
    )
    
    # ===== Gate Analysis =====
    results["gates"] = {}
    for task in ["tag", "time", "scale", "negative"]:
        gates = np.array(all_predictions[task]["gate"])
        results["gates"][task] = {
            "mean": float(gates.mean()),
            "std": float(gates.std()),
            "min": float(gates.min()),
            "max": float(gates.max()),
            "median": float(np.median(gates))
        }
    
    # ===== Save Error Analysis CSVs =====
    for task, errors in errors_by_category.items():
        if errors:
            df = pd.DataFrame(errors)
            error_path = os.path.join(output_dir, f"{run_name}_errors_{task}.csv")
            df.to_csv(error_path, index=False)
            print(f"✅ Saved {len(errors)} {task} errors to {error_path}")
    
    # ===== Save Overall Results =====
    results_path = os.path.join(output_dir, f"{run_name}_results.json")
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved results to {results_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for task in ["tag", "time", "scale", "negative"]:
        print(f"\n{task.upper()}:")
        print(f"  Macro F1: {results[task].get('macro_f1', 0):.4f}")
        if "hits@3" in results.get(task, {}):
            print(f"  Hits@3: {results[task]['hits@3']:.4f}")
            print(f"  Hits@5: {results[task]['hits@5']:.4f}")
        print(f"  Gate Mean: {results['gates'][task]['mean']:.4f}")
    
    return results


def load_model_config(config_path):
    """
    Load model configuration saved during training.
    
    Args:
        config_path (str): Path to model_config.json
        
    Returns:
        dict: Configuration dictionary
        
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


def get_device():
    """
    Get the appropriate device (GPU or CPU).
    
    Returns:
        str: "cuda" if available, else "cpu"
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device
