"""
Example Evaluation Script for SecBERT Model

This script demonstrates how to:
1. Load trained model weights and configuration
2. Load test data
3. Run inference and evaluation
4. Generate error analysis reports

Usage:
    python evaluate_example.py --model-path model_weight/secbert/secbert-0426-173_step820000.pt \
        --config-path model_weight/secbert/secbert-0426-173_config.json \
        --test-files processed_data_task1_smaller/test_50k.jsonl \
        --output-dir result/secbert
"""

import os
import sys
import argparse
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import from the module
sys.path.insert(0, os.path.dirname(__file__))

from model import MultiTaskModel
from preprocess import MultiTaskIterableDataset
from evaluate import evaluate_model, load_model_config, get_device


def main(args):
    """Main evaluation function."""
    
    # ===== Setup =====
    device = get_device()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ===== Load Configuration =====
    print("Loading model configuration...")
    config = load_model_config(args.config_path)
    
    # ===== Prepare Mappings =====
    tag2id = config["tag2id"]
    id2tag = {int(k): v for k, v in config["id2tag"].items()}
    
    time2id = config["time2id"]
    id2time = {int(k): v for k, v in config["id2time"].items()}
    
    scale2id = config["scale2id"]
    id2scale = {int(k): v for k, v in config["id2scale"].items()}
    
    # ===== Load Tokenizer =====
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/sec-bert-base")
    
    # ===== Initialize Model =====
    print("Initializing model...")
    model_args = config["model_args"]
    model = MultiTaskModel(
        bert_model_name=model_args["bert_model_name"],
        num_tags=model_args["num_tags"],
        num_times=model_args["num_times"],
        num_scales=model_args["num_scales"]
    )
    
    # ===== Load Trained Weights =====
    print(f"Loading model weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    print("✅ Model loaded successfully")
    
    # ===== Create Test Data Loader =====
    print("Loading test data...")
    target_attrs = ["tag", "time", "scale", "negative", "fact"]
    
    test_dataset = MultiTaskIterableDataset(
        files=args.test_files.split(','),
        target_attrs=target_attrs,
        tokenizer=tokenizer,
        tag2id=tag2id,
        time2id=time2id,
        scale2id=scale2id
    )
    
    def collate_fn(batch):
        """Custom collate function for IterableDataset batches."""
        result = {}
        for key in batch[0].keys():
            if key in ["input_ids", "attention_mask", "start_token", "end_token", "value",
                      "tag", "time", "scale", "negative", "fact"]:
                result[key] = torch.stack([
                    torch.tensor(item[key]) if not isinstance(item[key], torch.Tensor) 
                    else item[key] 
                    for item in batch
                ])
            else:
                result[key] = [item[key] for item in batch]
        return result
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=device == "cuda"
    )
    
    print(f"✅ Test data loaded (batch size: {args.batch_size})")
    
    # ===== Run Evaluation =====
    print("\nRunning evaluation...")
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        id2tag=id2tag,
        id2time=id2time,
        id2scale=id2scale,
        output_dir=args.output_dir,
        run_name=args.run_name,
        classification_missing_value=config["classification_missing_value"],
        numeric_missing_value=config["numeric_missing_value"]
    )
    
    # ===== Print Detailed Results =====
    print("\n" + "="*70)
    print("DETAILED EVALUATION RESULTS")
    print("="*70)
    
    for task in ["tag", "time", "scale", "negative"]:
        print(f"\n{'='*70}")
        print(f"{task.upper()} CLASSIFICATION")
        print(f"{'='*70}")
        
        task_results = results.get(task, {})
        
        if "error" in task_results:
            print(f"  ❌ Error: {task_results['error']}")
            continue
        
        print(f"\n  📊 Overall Metrics:")
        print(f"    Macro Precision: {task_results.get('macro_precision', 0):.4f}")
        print(f"    Macro Recall:    {task_results.get('macro_recall', 0):.4f}")
        print(f"    Macro F1:        {task_results.get('macro_f1', 0):.4f}")
        print(f"    Micro Precision: {task_results.get('micro_precision', 0):.4f}")
        print(f"    Micro Recall:    {task_results.get('micro_recall', 0):.4f}")
        print(f"    Micro F1:        {task_results.get('micro_f1', 0):.4f}")
        
        if "hits@3" in task_results:
            print(f"\n  🎯 Ranking Metrics:")
            print(f"    Hits@3: {task_results.get('hits@3', 0):.4f}")
            print(f"    Hits@5: {task_results.get('hits@5', 0):.4f}")
        
        # Gate statistics
        gate_stats = results.get("gates", {}).get(task, {})
        if gate_stats:
            print(f"\n  🚪 Gate Statistics (Noise Detection):")
            print(f"    Mean Gate Value: {gate_stats.get('mean', 0):.4f}")
            print(f"    Std Dev:         {gate_stats.get('std', 0):.4f}")
            print(f"    Min:             {gate_stats.get('min', 0):.4f}")
            print(f"    Max:             {gate_stats.get('max', 0):.4f}")
            print(f"    Median:          {gate_stats.get('median', 0):.4f}")
        
        # Top errors (if available)
        per_class = task_results.get("per_class", {})
        if per_class:
            print(f"\n  📈 Per-Class Performance (Top 5 by support):")
            sorted_classes = sorted(
                per_class.items(),
                key=lambda x: x[1].get("support", 0),
                reverse=True
            )[:5]
            for class_name, metrics in sorted_classes:
                print(f"    {class_name:<30} F1={metrics.get('f1', 0):.4f} "
                      f"(support={metrics.get('support', 0)})")
    
    print(f"\n{'='*70}")
    print("✅ Evaluation complete!")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SecBERT Multi-Task Model")
    
    # Model paths
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model weights (.pt file)")
    parser.add_argument("--config-path", type=str, required=True,
                       help="Path to model configuration JSON")
    
    # Data
    parser.add_argument("--test-files", type=str, default="processed_data_task1_smaller/test_50k.jsonl",
                       help="Comma-separated list of test JSONL files")
    
    # Output
    parser.add_argument("--output-dir", type=str, default="result/secbert",
                       help="Directory to save evaluation results")
    parser.add_argument("--run-name", type=str, default="secbert-0426-173",
                       help="Run name for result files")
    
    # Evaluation parameters
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for inference")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.config_path):
        print(f"❌ Error: Config file not found: {args.config_path}")
        sys.exit(1)
    
    main(args)
