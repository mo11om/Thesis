import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import faiss  # Requires: pip install faiss-cpu or faiss-gpu

# Import from your existing utils
from secbert_utils import (
    set_seed, ProcessedIterableDataset, MultiTaskModel, 
    compute_loss, CB_CE_Loss, save_checkpoint, validate_model,
    CLASSIFICATION_MISSING_VALUE
)

# ============================================================================
# SSR Framework Functions
# ============================================================================

def extract_features_and_labels(model, dataloader, device):
    """Extracts [CLS] embeddings, current labels, and model predictions for SSR."""
    model.eval()
    all_features = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="SSR Phase 1: Extracting Features"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["tag"].to(device)
            
            # Forward pass through BERT backbone
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            
            # Use [CLS] token representation for SSR topological distance
            features = outputs.last_hidden_state[:, 0, :] 
            features = F.normalize(features, p=2, dim=1)
            
            # Get PMC predictions for the tag task
            logits = model.tag_head(features)
            
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())
            all_logits.append(logits.cpu())
            
    return torch.cat(all_features), torch.cat(all_labels), torch.cat(all_logits)

def ssr_relabel_and_select(features, labels, logits, theta_r=0.9, theta_s=1.0, k=50, num_classes=978):
    """
    Implements PMC Relabelling and NPK Sample Selection with optimized FAISS.
    """
    import numpy as np
    import faiss
    
    N = features.size(0)
    probs = F.softmax(logits, dim=1)
    max_probs, preds = torch.max(probs, dim=1)
    
    # --- 1. Relabelling Mechanism (PMC thresholding) ---
    new_labels = labels.clone()
    relabel_mask = max_probs > theta_r
    new_labels[relabel_mask] = preds[relabel_mask]
    
    print(f"SSR Relabelled {relabel_mask.sum().item()} noisy samples.")
    
    # Calculate class distribution \pi for balancing
    valid_mask = new_labels != -100 # CLASSIFICATION_MISSING_VALUE
    label_counts = torch.bincount(new_labels[valid_mask], minlength=num_classes).float()
    pi = label_counts / label_counts.sum()
    pi[pi == 0] = 1e-8 # Prevent division by zero
    
    # --- 2. k-NN Density Calculation (NPK) using FAISS ---
    print("SSR Computing k-NN for sample selection (GPU Accelerated)...")
    
    # Free up PyTorch VRAM before allocating FAISS buffers
    torch.cuda.empty_cache() 
    
    d = features.shape[1]
    faiss_features = features.numpy()
    
    # Initialize CPU Index
    cpu_index = faiss.IndexFlatIP(d)
    
    # Move Index to GPU (Requires faiss-gpu)
    try:
        res = faiss.StandardGpuResources()
        # Allocate less memory for temporary resources to prevent OOM
        res.setTempMemory(512 * 1024 * 1024) 
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        print("Successfully moved FAISS index to GPU.")
    except Exception as e:
        print(f"Could not use GPU for FAISS: {e}. Falling back to CPU. This will take a few minutes!")
        index = cpu_index 

    # Add features to the index
    index.add(faiss_features)
    
    # THE FIX: Dramatically lower the batch size to prevent VRAM overflow
    # 256 queries * 728,000 DB size * 4 bytes = ~745 MB VRAM footprint
    search_batch_size = 16 
    all_indices = []
    
    print(f"Searching for nearest neighbors in batches of {search_batch_size}...")
    for i in tqdm(range(0, N, search_batch_size), desc="FAISS Search"):
        end_idx = min(i + search_batch_size, N)
        batch_query = faiss_features[i:end_idx]
        
        # Search for k+1 because the sample itself will be the 1st result
        _, batch_indices = index.search(batch_query, k + 1) 
        all_indices.append(batch_indices)
        
    indices = np.vstack(all_indices)
    indices = torch.tensor(indices[:, 1:]) # Drop self-reference (the 0th column)
    
    # --- 3. Calculate neighborhood label distribution ---
    selected_mask = torch.zeros(N, dtype=torch.bool)
    
    for i in tqdm(range(N), desc="SSR Phase 2: Selecting Clean Samples"):
        if new_labels[i] == -100:
            continue
            
        neighbor_labels = new_labels[indices[i]]
        valid_neighbors = neighbor_labels[neighbor_labels != -100]
        
        if len(valid_neighbors) == 0:
            continue
            
        q_i_prime = torch.bincount(valid_neighbors, minlength=num_classes).float() / len(valid_neighbors)
        
        # Balance the distribution and calculate consistency
        q_i = q_i_prime / pi
        c_i = q_i[new_labels[i]] / (q_i.max() + 1e-8)
        
        if c_i >= theta_s:
            selected_mask[i] = True

    selected_indices = torch.nonzero(selected_mask).squeeze()
    print(f"SSR Selected {len(selected_indices)} / {N} samples as clean.")
    
    return selected_indices, new_labels
# def ssr_relabel_and_select(features, labels, logits, theta_r=0.9, theta_s=1.0, k=50, num_classes=978):
#     """
#     Implements PMC Relabelling and NPK Sample Selection with optimized FAISS.
#     """
#     import numpy as np
    
#     N = features.size(0)
#     probs = F.softmax(logits, dim=1)
#     max_probs, preds = torch.max(probs, dim=1)
    
#     # --- 1. Relabelling Mechanism (PMC thresholding) ---
#     new_labels = labels.clone()
#     relabel_mask = max_probs > theta_r
#     new_labels[relabel_mask] = preds[relabel_mask]
    
#     print(f"SSR Relabelled {relabel_mask.sum().item()} noisy samples.")
    
#     # Calculate class distribution \pi for balancing
#     valid_mask = new_labels != -100 # CLASSIFICATION_MISSING_VALUE
#     label_counts = torch.bincount(new_labels[valid_mask], minlength=num_classes).float()
#     pi = label_counts / label_counts.sum()
#     pi[pi == 0] = 1e-8 # Prevent division by zero
    
#     # --- 2. k-NN Density Calculation (NPK) using FAISS ---
#     print("SSR Computing k-NN for sample selection (GPU Accelerated)...")
    
#     import numpy as np
    
#     d = features.shape[1]
    
#     # FIX 1: Absolutely force float32 and C-contiguous memory layout
#     # This prevents the CUBLAS memory access violation (Error 13)
#     faiss_features = np.ascontiguousarray(features.cpu().numpy(), dtype=np.float32)
    
#     # Initialize CPU Index
#     cpu_index = faiss.IndexFlatIP(d)
    
#     # Move Index to GPU 
#     try:
#         res = faiss.StandardGpuResources()
        
#         # FIX 2: Explicitly assign a larger temporary memory workspace for CUBLAS
#         # Allocate 512 MB (512 * 1024 * 1024 bytes) to handle the large distance matrix
#         res.setTempMemory(536870912) 
        
#         index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
#         print("Successfully moved FAISS index to GPU.")
#     except Exception as e:
#         print(f"faiss-gpu exception: {e}. Falling back to CPU.")
#         index = cpu_index 

#     # Add features to the index
#     index.add(faiss_features)
    
#     # Batched Search
#     # FIX 3: Keep batch size conservative (1024 or 2048) to prevent VRAM spikes
#     search_batch_size = 1024 
#     all_indices = []
    
#     print("Searching for nearest neighbors in batches...")
#     for i in tqdm(range(0, N, search_batch_size), desc="FAISS Search"):
#         end_idx = min(i + search_batch_size, N)
        
#         # Ensure the query chunk itself is also contiguous float32
#         batch_query = np.ascontiguousarray(faiss_features[i:end_idx], dtype=np.float32)
        
#         # Search for k+1 because the sample itself will be the 1st result
#         _, batch_indices = index.search(batch_query, k + 1) 
#         all_indices.append(batch_indices)
        
#     indices = np.vstack(all_indices)
#     indices = torch.tensor(indices[:, 1:]) # Drop self-reference (the 0th column)
#     # --- 3. Calculate neighborhood label distribution ---
#     selected_mask = torch.zeros(N, dtype=torch.bool)
    
#     for i in tqdm(range(N), desc="SSR Phase 2: Selecting Clean Samples"):
#         if new_labels[i] == -100:
#             continue
            
#         neighbor_labels = new_labels[indices[i]]
#         valid_neighbors = neighbor_labels[neighbor_labels != -100]
        
#         if len(valid_neighbors) == 0:
#             continue
            
#         q_i_prime = torch.bincount(valid_neighbors, minlength=num_classes).float() / len(valid_neighbors)
        
#         # Balance the distribution and calculate consistency
#         q_i = q_i_prime / pi
#         c_i = q_i[new_labels[i]] / (q_i.max() + 1e-8)
        
#         if c_i >= theta_s:
#             selected_mask[i] = True

#     selected_indices = torch.nonzero(selected_mask).squeeze()
#     print(f"SSR Selected {len(selected_indices)} / {N} samples as clean.")
    
#     return selected_indices, new_labels


# def ssr_relabel_and_select(features, labels, logits, theta_r=0.9, theta_s=1.0, k=50, num_classes=978):
#     """
#     Implements PMC Relabelling and NPK Sample Selection.
#     """
#     N = features.size(0)
#     probs = F.softmax(logits, dim=1)
#     max_probs, preds = torch.max(probs, dim=1)
    
#     # --- 1. Relabelling Mechanism (PMC thresholding) ---
#     new_labels = labels.clone()
#     relabel_mask = max_probs > theta_r
#     new_labels[relabel_mask] = preds[relabel_mask]
    
#     print(f"SSR Relabelled {relabel_mask.sum().item()} noisy samples.")
    
#     # Calculate class distribution \pi for balancing
#     valid_mask = new_labels != CLASSIFICATION_MISSING_VALUE
#     label_counts = torch.bincount(new_labels[valid_mask], minlength=num_classes).float()
#     pi = label_counts / label_counts.sum()
#     pi[pi == 0] = 1e-8 # Prevent division by zero
    
#     # --- 2. k-NN Density Calculation (NPK) using FAISS ---
#     print("SSR Computing k-NN for sample selection...")
#     index = faiss.IndexFlatIP(features.shape[1]) # Inner product of normalized vectors = Cosine Sim
#     faiss_features = features.numpy()
#     index.add(faiss_features)
#     distances, indices = index.search(faiss_features, k + 1)
#     indices = torch.tensor(indices[:, 1:]) # Drop self-reference
    
#     # Calculate neighborhood label distribution
#     selected_mask = torch.zeros(N, dtype=torch.bool)
    
#     for i in tqdm(range(N), desc="SSR Phase 2: Selecting Clean Samples"):
#         if new_labels[i] == CLASSIFICATION_MISSING_VALUE:
#             continue
            
#         neighbor_labels = new_labels[indices[i]]
#         valid_neighbors = neighbor_labels[neighbor_labels != CLASSIFICATION_MISSING_VALUE]
        
#         if len(valid_neighbors) == 0:
#             continue
            
#         q_i_prime = torch.bincount(valid_neighbors, minlength=num_classes).float() / len(valid_neighbors)
        
#         # Balance the distribution and calculate consistency
#         q_i = q_i_prime / pi
#         c_i = q_i[new_labels[i]] / (q_i.max() + 1e-8)
        
#         if c_i >= theta_s:
#             selected_mask[i] = True

#     selected_indices = torch.nonzero(selected_mask).squeeze()
#     print(f"SSR Selected {len(selected_indices)} / {N} samples as clean.")
    
#     return selected_indices, new_labels


def ssr_consistency_loss(h1, h2):
    """SSR+ Feature consistency loss using cosine similarity."""
    h1 = F.normalize(h1, p=2, dim=-1)
    h2 = F.normalize(h2, p=2, dim=-1)
    return -(h1 * h2).sum(dim=-1).mean()

# ============================================================================
# Main Training Script
# ============================================================================

def main():
    # --- Configuration ---
    set_seed(3047)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "secbert"
    date = "0426"
    index = "SSR_173"
    run_name = f"{model_name}-{date}-{index}"
    
    batch_size = 64
    num_epochs = 30
    eval_step = 20000 
    patience = 3
    
    # SSR Hyperparameters
    num_warmup_epochs = 1
    theta_r = 0.9    # Relabelling threshold
    theta_s = 1.0    # Selection threshold
    knn_k = 50       # Neighbors for NPK
    lambda_fc = 1.0  # Weight for feature consistency loss

    # Loss Configuration (Mock lists - replace with your actual lists from your notebook)
    tag_list = [f"tag_{i}" for i in range(978)]  # Update to your actual len
    time_list = [f"time_{i}" for i in range(9)]  # Update to your actual len
    scale_list = [f"scale_{i}" for i in range(25)] # Update to your actual len
    
    num_tag_samples = [1] * len(tag_list) # Update with actual loaded counts
    num_time_samples = [1] * len(time_list) # Update with actual loaded counts
    
    tag_loss_fn = CB_CE_Loss(num_tag_samples, beta=0.99)
    time_loss_fn = CB_CE_Loss(num_time_samples)
    # Define scale_loss_fn and neg_loss_fn as in your notebook...
    scale_loss_fn = nn.CrossEntropyLoss(ignore_index=CLASSIFICATION_MISSING_VALUE)
    neg_loss_fn = nn.CrossEntropyLoss(ignore_index=CLASSIFICATION_MISSING_VALUE)

    # --- Data Loading ---
    train_files = ["processed_iterable_dataset/train_400k.jsonl"]
    valid_files = ["processed_iterable_dataset/valid_50k.jsonl"] 
    
    train_dataset = ProcessedIterableDataset(train_files)
    valid_dataset = ProcessedIterableDataset(valid_files)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4, drop_last=False)
    
    train_approx_batches = 11387 # From your notebook

    # --- Model & Optimizer Initialization ---
    model = MultiTaskModel(
        "nlpaueb/sec-bert-base",
        num_tags=len(tag_list), 
        num_times=len(time_list),
        num_scales=len(scale_list)
    ).to(device)

    optimizer = AdamW([
        {"params": model.bert.parameters(), "lr": 1e-5, "weight_decay": 1e-2},  
        {"params": model.tag_head.parameters(), "lr": 5e-4,  "weight_decay": 1e-2},  
        {"params": model.time_head.parameters(), "lr": 1e-5,  "weight_decay": 1e-2},  
        {"params": model.scale_head.parameters(), "lr": 3e-5,  "weight_decay": 1e-2},
        {"params": model.negative_head.parameters(), "lr": 2e-5,  "weight_decay": 1e-2},  
    ])

    num_total_steps = num_epochs * train_approx_batches
    scheduler = CosineAnnealingLR(optimizer, T_max=num_total_steps // 4, eta_min=1e-7)

    # --- Weights & Biases ---
    wandb.init(project="multi-task-model", name=run_name, config={"epochs": num_epochs})

    # --- Training Loop ---
    best_val_loss = float("inf")
    early_stop_counter = 0 
    step = 0
    progress_bar = tqdm(range(num_total_steps), desc="Training", dynamic_ncols=True)

    for epoch in range(num_epochs):
        if early_stop_counter >= patience:
            print("Early stopping triggered. Training stopped.")
            break
            
        task_weights = {"tag": 10.0, "time": 0.3, "scale": 0.2, "negative": 10.0}
        
        # ---------------------------------------------------------
        # SSR DYNAMIC SELECTION & RELABELLING
        # ---------------------------------------------------------
        if epoch >= num_warmup_epochs:
            print(f"\n--- Initiating SSR Robust Learning Phase for Epoch {epoch} ---")
            features, labels, logits = extract_features_and_labels(model, train_loader, device)
            selected_indices, relabelled_tags = ssr_relabel_and_select(
                features, labels, logits, 
                theta_r=theta_r, theta_s=theta_s, k=knn_k, num_classes=len(tag_list)
            )
            selected_set = set(selected_indices.tolist())
        else:
            print(f"\n--- Warmup Phase (Standard CE) Epoch {epoch} ---")
            selected_set = None
            relabelled_tags = None

        # ---------------------------------------------------------
        # BATCH TRAINING
        # ---------------------------------------------------------
        model.train()
        train_loss = 0
        global_idx = 0 # Tracks sequential index for IterableDataset matching
        
        for batch in train_loader:
            batch_size_current = batch["input_ids"].size(0)
            batch_indices = list(range(global_idx, global_idx + batch_size_current))
            global_idx += batch_size_current
            
            # Apply SSR Filtering if robust phase is active
            # Apply SSR Filtering if robust phase is active
            if selected_set is not None:
                clean_mask = torch.tensor([i in selected_set for i in batch_indices])
                if not clean_mask.any():
                    continue # Skip batch if all samples are noisy
                
                # Relabel the tag targets for the current batch
                batch["tag"] = relabelled_tags[batch_indices]
                
                # Filter out noisy samples from the batch safely based on type
                clean_mask_list = clean_mask.tolist()
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        # Safely mask PyTorch tensors (input_ids, attention_mask, tag, etc.)
                        batch[key] = batch[key][clean_mask]
                    elif isinstance(batch[key], (list, tuple)):
                        # Safely filter Python lists/tuples (context, doc_link, etc.)
                        filtered_items = [item for item, keep in zip(batch[key], clean_mask_list) if keep]
                        # Preserve original type (list or tuple)
                        batch[key] = type(batch[key])(filtered_items)

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

            # Forward pass 1 (Standard Supervised)
            outputs = model(input_ids, attention_mask, start_tokens, end_tokens)
            loss, losses = compute_loss(outputs, targets, values, task_weights,
                                        tag_loss_fn=tag_loss_fn, time_loss_fn=time_loss_fn,
                                        scale_loss_fn=scale_loss_fn, neg_loss_fn=neg_loss_fn)
            
            # Forward pass 2 (SSR+ Feature Consistency)
            if epoch >= num_warmup_epochs:
                # The identical inputs will trigger different dropout masks in SecBERT
                outputs_view2 = model(input_ids, attention_mask, start_tokens, end_tokens)
                h1 = outputs["tag"] 
                h2 = outputs_view2["tag"]
                
                loss_fc = ssr_consistency_loss(h1, h2)
                loss = loss + (lambda_fc * loss_fc)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()
            step += 1
            epoch_progress = step / train_approx_batches
            
            progress_bar.set_postfix(loss=loss.item(), epoch=f"{epoch_progress:.2f}")
            progress_bar.update(1)

            # ---------------------------------------------------------
            # VALIDATION & CHECKPOINTING (Simplified)
            # ---------------------------------------------------------
            if step % eval_step == 0:
                os.makedirs(f'check_point/{model_name}', exist_ok=True)
                save_checkpoint(model, optimizer, scheduler, epoch, step, f'check_point/{model_name}/{date}_{index}_step{step}.pth')
                
                val_loss, val_losses = validate_model(model, valid_loader, task_weights, device, 
                                                      tag_loss_fn, time_loss_fn, scale_loss_fn, neg_loss_fn)
                model.train()
                
                wandb.log({"val_loss": val_loss, "step": step})
                print(f"\nStep {step} Validation Loss = {val_loss:.4f}")
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    early_stop_counter = 0
                    torch.save(model.state_dict(), f"model_weight/{model_name}/{date}_{index}_best.pt")
                else:
                    early_stop_counter += 1

    wandb.finish()

if __name__ == "__main__":
    main()