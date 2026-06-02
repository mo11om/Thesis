import os
import json
import torch
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import reusable functions and classes from shared_utils
from shared_utils import (
    ProcessedIterableDataset, 
    MultiTaskModel,
    CB_CE_Loss, 
    FocalLoss, 
    load_model_config,
    evaluate_classification, 
    evaluate_model
)

def load_counter_data(target_attr, counter_dir='processed_iterable_dataset_modern/counter'):
    """Helper to load loss weighting counters from the modern dataset directory."""
    target_path = os.path.join(counter_dir, f'secbert_train_small_{target_attr}.json')
    if not os.path.exists(target_path):
        print(f"⚠️ Warning: Counter file not found at {target_path}")
        return {}
    with open(target_path, "r", encoding='utf-8') as f:
        return json.load(f)

def main():
    # ============================================================================
    # ===== DYNAMIC CONFIGURATION =====
    # ============================================================================
    # Update these variables to match your specific BGE training run
    RUN_NAME = "bge-0601-173" 
    MODEL_NAME = "bge"
    RUN_DATE = "0601"
    RUN_INDEX = 173
    CHECKPOINT_STEP = 90001 # Update to your actual checkpoint step
    CLASSIFICATION_MISSING_VALUE = -100
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        # Fallback configuration tailored to your new directory structure
        config = {
            'data': {
                'batch_size': 256, 
                'num_workers': 10, 
                # Evaluated on the test set from your ls output
                'test_files': ["processed_iterable_dataset_bge/test_50k.jsonl"] 
            },
            'loss': {
                'time': {'min_weight': 1.0, 'special_weights': {0: 1.5, 6: 1.5}},
                'scale': {'weights': {15: 1.2, 21: 1.5}},
                'negative': {'alpha': 0.25, 'gamma': 3.0}
            }
        }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    # ============================================================================
    # ===== LOAD MODEL CONFIGURATION =====
    # ============================================================================
    config_path = f'model_weight/{MODEL_NAME}/{RUN_NAME}_config.json'
    model_config = load_model_config(config_path)
    model_args = model_config["model_args"]
    
    # Extract mappings directly from the trained model config 
    tag2id = model_config.get("tag2id", {})
    id2scale = model_config.get("id2scale", {str(idx): str(idx - 12) for idx in range(25)})
    scale_list = [str(i) for i in range(-12, 13)]

    # Initialize BGE tokenizer dynamically
    bert_model_name = model_args.get("bert_model_name", "BAAI/bge-base-en-v1.5")
    print(f"⚙️ Loading tokenizer: {bert_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

    # ============================================================================
    # ===== DATASET SETUP =====
    # ============================================================================
    # Use test_50k.jsonl from your processed_iterable_dataset_bge folder
    test_files = config['data'].get('test_files', ["processed_iterable_dataset_bge/test_50k_filtered.jsonl"])
    batch_size = config['data'].get('batch_size', 256)
    num_workers = config['data'].get('num_workers', 4)

    print(f"📂 Loading evaluation data from: {test_files}")
    test_dataset = ProcessedIterableDataset(test_files)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    # ============================================================================
    # ===== LOSS FUNCTIONS SETUP =====
    # ============================================================================
    train_time_counts = load_counter_data("time", counter_dir='processed_iterable_dataset_bge/counter')
    train_time_counts.pop('-100', None)
    
    train_tag_counts = load_counter_data("tag", counter_dir='processed_iterable_dataset_bge/counter')
    train_tag_counts.pop('-100', None)
    num_tag_samples = [train_tag_counts.get(str(tag2id.get(tag, 0)), 0) for tag in tag2id.keys()]

    # Time Loss Weights
    if train_time_counts:
        total_samples = sum(train_time_counts.values())
        num_classes = len(train_time_counts)
        time_class_weights = {int(cls): total_samples / (num_classes * count) for cls, count in train_time_counts.items()}
        time_class_weights = [time_class_weights.get(i, 1.0) for i in range(len(time_class_weights))]
        time_smoothed_weights = np.clip(np.log1p(time_class_weights), config['loss']['time']['min_weight'], None)
        
        # Apply special weights from config
        for idx, weight in config['loss']['time']['special_weights'].items():
            if int(idx) < len(time_smoothed_weights):
                time_smoothed_weights[int(idx)] = weight
        time_class_weights_tensor = torch.tensor(time_smoothed_weights, dtype=torch.float).to(device)
    else:
        time_class_weights_tensor = None

    # Scale Loss Weights
    scale_class_weights_dict = {int(k): v for k, v in config['loss']['scale']['weights'].items()}
    weights_list = [scale_class_weights_dict.get(i, 1.0) for i in range(len(scale_list))]
    scale_class_weights_tensor = torch.tensor(weights_list, dtype=torch.float).to(device)

    # Initialize Loss Modules
    tag_loss_fn = CB_CE_Loss(num_tag_samples, beta=0.99, ignore_index=CLASSIFICATION_MISSING_VALUE) if num_tag_samples else None
    time_loss_fn = nn.CrossEntropyLoss(weight=time_class_weights_tensor, ignore_index=CLASSIFICATION_MISSING_VALUE, reduction='none')
    scale_loss_fn = nn.CrossEntropyLoss(weight=scale_class_weights_tensor, ignore_index=CLASSIFICATION_MISSING_VALUE, reduction='none')
    neg_loss = FocalLoss(alpha=config['loss']['negative']['alpha'], gamma=config['loss']['negative']['gamma'], reduction="none")

    # ============================================================================
    # ===== MODEL INITIALIZATION =====
    # ============================================================================
    model = MultiTaskModel(
        bert_model_name=bert_model_name,
        num_tags=model_args["num_tags"],
        num_times=model_args["num_times"],
        num_scales=model_args["num_scales"]
    )

    checkpoint_path = os.path.join("model_weight", "bge", f"{RUN_DATE}_{RUN_INDEX}_step{CHECKPOINT_STEP}.pt")
    print(f"🔄 Loading model checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Model checkpoint not found: {checkpoint_path}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # ============================================================================
    # ===== EVALUATION =====
    # ============================================================================
    base_error_dir = "error_analysis"
    os.makedirs(base_error_dir, exist_ok=True)
    error_file_path = os.path.join(base_error_dir, f"{MODEL_NAME}_{RUN_DATE}_{RUN_INDEX}")

    task_weights = {"tag": 10.0, "time": 0.3, "scale": 0.2, "negative": 10.0}
    print(f"🔍 Running inference with {MODEL_NAME} model (checkpoint: step {CHECKPOINT_STEP})")

    print("\n" + "="*70)
    print("STARTING EVALUATION")
    print("="*70)

    test_all_losses, test_predictions, test_ground_truths, avg_hits_k, gate_df = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        task_weights=task_weights,
        error_file_path=error_file_path,
        # id2scale=id2scale,
        save_errors=True,
        verbose=True,
        tag_loss_fn=tag_loss_fn,
        time_loss_fn=time_loss_fn,
        scale_loss_fn=scale_loss_fn,
        neg_loss=neg_loss
    )

    # ============================================================================
    # ===== SAVE RESULTS =====
    # ============================================================================
    os.makedirs("result", exist_ok=True)
    gate_df.to_csv("result/cached_gate_df.csv", index=False)
    print("✅ Cached gate_df to result/cached_gate_df.csv")

    attrs = ["scale", "negative", "tag", "time", "fact"]
    for attr in attrs:
        if attr in test_predictions and attr in test_ground_truths:
            df = pd.DataFrame({
                f"true_{attr}": test_ground_truths[attr],
                f"pred_{attr}": test_predictions[attr],
            })
            result_path = os.path.join("result", f"{MODEL_NAME}_{RUN_DATE}_{RUN_INDEX}_result_{attr}.csv")
            df.to_csv(result_path, index=False, encoding="utf-8")
            print(f"✅ Saved {attr} results to {result_path} ({len(df)} samples)")
            
            # Print basic classification metrics
            if attr != "fact":
                evaluate_classification(test_ground_truths[attr], test_predictions[attr], attr)
    
    print(f"\n✅ BGE inference complete! Results saved to result/")

if __name__ == "__main__":
    main()