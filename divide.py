# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import faiss
import numpy as np
import json
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Assuming these are available in your environment from secbert_utils
from secbert_utils import (
    ProcessedIterableDataset, MultiTaskModel, evaluate_model,
    CLASSIFICATION_MISSING_VALUE
)

# ============================================================================
# 1. Feature Extraction & NPK Selection Mechanism
# ============================================================================

def extract_features_and_labels(model, dataloader, device):
    """Extracts [CLS] embeddings and current labels for SSR selection."""
    model.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting Features"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["tag"]
            
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            
            # Use [CLS] token representation and L2 normalize
            features = outputs.last_hidden_state[:, 0, :] 
            features = F.normalize(features, p=2, dim=1)
            
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            
    return torch.cat(all_features), torch.cat(all_labels)

def get_clean_subset(features, labels, k=50, theta_s=1.0, num_classes=978):
    """
    NPK Selection using FAISS. Calculates consistency c_i.
    Returns a set of integer indices of the clean samples.
    """
    N = features.size(0)
    d = features.size(1)

    valid_mask = labels != CLASSIFICATION_MISSING_VALUE
    label_counts = torch.bincount(labels[valid_mask], minlength=num_classes).float()
    pi = label_counts / label_counts.sum()
    pi[pi == 0] = 1e-8 

    print("Computing k-NN for sample selection (Batched FAISS)...")
    faiss_features = np.ascontiguousarray(features.numpy(), dtype=np.float32)
    cpu_index = faiss.IndexFlatIP(d)
    
    try:
        res = faiss.StandardGpuResources()
        res.setTempMemory(512 * 1024 * 1024) 
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    except Exception as e:
        print(f"FAISS GPU failed: {e}. Falling back to CPU.")
        index = cpu_index

    index.add(faiss_features)

    search_batch_size = 16 
    all_indices = []
    
    for i in tqdm(range(0, N, search_batch_size), desc="FAISS Search"):
        end_idx = min(i + search_batch_size, N)
        batch_query = np.ascontiguousarray(faiss_features[i:end_idx], dtype=np.float32)
        _, batch_indices = index.search(batch_query, k + 1) 
        all_indices.append(batch_indices)
        
    indices = np.vstack(all_indices)
    indices = torch.tensor(indices[:, 1:]) # Drop self-reference
    
    selected_mask = torch.zeros(N, dtype=torch.bool)
    
    for i in tqdm(range(N), desc="Selecting Clean Samples"):
        if labels[i] == CLASSIFICATION_MISSING_VALUE:
            continue
            
        neighbor_labels = labels[indices[i]]
        valid_neighbors = neighbor_labels[neighbor_labels != CLASSIFICATION_MISSING_VALUE]
        
        if len(valid_neighbors) == 0:
            continue
            
        q_i_prime = torch.bincount(valid_neighbors, minlength=num_classes).float() / len(valid_neighbors)
        q_i = q_i_prime / pi
        c_i = q_i[labels[i]] / (q_i.max() + 1e-8)
        
        if c_i >= theta_s:
            selected_mask[i] = True

    selected_indices = torch.nonzero(selected_mask).squeeze()
    print(f"Identified {len(selected_indices)} / {N} clean samples.")
    return set(selected_indices.tolist())

# ============================================================================
# 2. File Creation & Metrics
# ============================================================================

def save_clean_dataset(original_file, clean_indices, output_file):
    """Reads the original JSONL and writes only the clean indices to a new file."""
    print(f"\nSaving clean dataset to: {output_file}")
    saved_count = 0
    with open(original_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for idx, line in enumerate(tqdm(f_in, desc="Writing File")):
            if idx in clean_indices:
                f_out.write(line)
                saved_count += 1
                
    print(f"Successfully saved {saved_count} lines to {output_file}")

def calculate_detailed_metrics(y_true, y_pred, attr="tag"):
    """Calculates Accuracy, Weighted F1, and Macro F1 ignoring missing values."""
    valid_indices = [i for i, t in enumerate(y_true) if t != CLASSIFICATION_MISSING_VALUE]
    y_true = [y_true[i] for i in valid_indices]
    y_pred = [y_pred[i] for i in valid_indices]

    if not y_true:
        return {"accuracy": 0, "macro_f1": 0, "weighted_f1": 0}

    acc = accuracy_score(y_true, y_pred)
    
    # Macro metrics
    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    # Weighted metrics
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1
    }

def print_evaluation_report(title, losses, hits_k, predictions, ground_truths):
    print(f"\n{'='*50}")
    print(f" {title} ")
    print(f"{'='*50}")
    print(f"Tag Loss       : {losses.get('tag', 0):.4f}")
    print(f"Tag Hits@1     : {hits_k.get('hits_1', 0):.4f}")
    print(f"Tag Hits@3     : {hits_k.get('hits_3', 0):.4f}")
    print(f"Tag Hits@5     : {hits_k.get('hits_5', 0):.4f}")
    print(f"--------------------------------------------------")
    
    # Calculate detailed classification metrics
    for task in ['tag', 'time', 'scale', 'negative']:
        if task in ground_truths and task in predictions:
            metrics = calculate_detailed_metrics(ground_truths[task], predictions[task], attr=task)
            print(f"[{task.upper()}] Accuracy     : {metrics['accuracy']:.4f}")
            print(f"[{task.upper()}] Weighted F1  : {metrics['weighted_f1']:.4f}")
            print(f"[{task.upper()}] Macro F1     : {metrics['macro_f1']:.4f}")
            print(f"- - - - - - - - - - - - - - - - - - - - - - - - - ")

# ============================================================================
# 3. Main Execution 
# ============================================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    num_tag_classes = 978 
    
    original_test_file = "processed_iterable_dataset/test_50k.jsonl"
    clean_test_file = "processed_iterable_dataset/test_50k_clean.jsonl"
    model_save_path = "model_weight/secbert/0426_SSR_173_best.pt" # Update path
    
    task_weights = {"tag": 10.0, "time": 0.3, "scale": 0.2, "negative": 10.0}

    print("--- 1. Initialization ---")
    model = MultiTaskModel(
        "nlpaueb/sec-bert-base", 
        num_tags=num_tag_classes, num_times=9, num_scales=25
    ).to(device)
    
    # Optional: If your weights are needed for feature extraction, load them here
    # model.load_state_dict(torch.load(model_save_path))
    
    # --- Phase A: Create the Clean Dataset ---
    print("\n--- 2. NPK Clean Subset Selection ---")
    # Must use num_workers=0 or 1 to ensure sequential line mapping
    feature_loader = DataLoader(ProcessedIterableDataset([original_test_file]), batch_size=batch_size, num_workers=0)
    features, labels = extract_features_and_labels(model, feature_loader, device)
    
    clean_indices = get_clean_subset(features, labels, k=50, theta_s=1.0, num_classes=num_tag_classes)
    
    # Save to the new file
    save_clean_dataset(original_test_file, clean_indices, clean_test_file)

    # # --- Phase B: Evaluation on ALL DATA ---
    # print("\n--- 3. Evaluating on ALL DATA ---")
    # all_loader = DataLoader(ProcessedIterableDataset([original_test_file]), batch_size=batch_size, num_workers=4)
    # all_losses, all_preds, all_gts, all_hits = evaluate_model(
    #     model=model, model_save_path=model_save_path, error_file_path=None, 
    #     test_loader=all_loader, task_weights=task_weights, device=device, save_errors=False
    # )
    # print_evaluation_report("EVALUATION: ALL TEST DATA", all_losses, all_hits, all_preds, all_gts)

    # # --- Phase C: Evaluation on CLEAN DATA ---
    # print("\n--- 4. Evaluating on CLEAN SUBSET ---")
    # clean_loader = DataLoader(ProcessedIterableDataset([clean_test_file]), batch_size=batch_size, num_workers=4)
    # cln_losses, cln_preds, cln_gts, cln_hits = evaluate_model(
    #     model=model, model_save_path=model_save_path, error_file_path=None, 
    #     test_loader=clean_loader, task_weights=task_weights, device=device, save_errors=False
    # )
    # print_evaluation_report("EVALUATION: CLEAN TEST SUBSET", cln_losses, cln_hits, cln_preds, cln_gts)


if __name__ == "__main__":
    main()

# %%



