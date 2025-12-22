# SecBERT Multi-Task Learning Framework

A modular, production-ready Python framework for training and evaluating a multi-task BERT model for financial information extraction from XBRL documents.

## Overview

**SecBERT** is a deep learning model that simultaneously performs multiple information extraction tasks:

1. **Tag Classification**: Identify XBRL financial tags (e.g., "us-gaap:Revenues")
2. **Time Classification**: Determine temporal dimension (instant/period, past/current/future)
3. **Scale Classification**: Extract magnitude/scale of numeric values (powers of 10, from -12 to +12)
4. **Negative Detection**: Binary classification for detecting negative values
5. **Gate-based Noise Detection**: Learn which samples are noisy for each task independently

### Architecture

```
Input Text → BERT Encoder → Target Span Aggregation → Task Heads
                                                    ↘
                                                      → Gates (per-task noise scores)
```

The model uses a **gating mechanism** to learn task-specific sample importance, allowing it to adaptively filter noisy training examples.

## Project Structure

```
model_code/
├── __init__.py                 # Package initialization
├── model.py                    # Neural network architecture
├── preprocess.py               # Data loading & tokenization
├── train.py                    # Training loop & loss functions
├── evaluate.py                 # Evaluation & metrics
├── train_example.py            # Complete training script
├── evaluate_example.py         # Complete evaluation script
└── README.md                   # This file
```

## Module Documentation

### 1. `model.py` - Neural Network Architecture

**Classes:**
- `GateHead`: Task-specific gating mechanism (outputs noise scores 0-1 for each task)
- `MultiTaskModel`: Main BERT-based model with 4 classification heads + gates

**Key Features:**
- Pre-trained BERT encoder with frozen/fine-tuned options
- Target span aggregation (averaging embeddings within identified spans)
- Task-specific heads with LayerNorm, GELU, and dropout for regularization
- Per-task gate heads that learn sample importance

**Example:**
```python
from model import MultiTaskModel

model = MultiTaskModel(
    bert_model_name="nlpaueb/sec-bert-base",
    num_tags=200,
    num_times=9,
    num_scales=25
)

outputs = model(
    input_ids=input_ids,           # (batch_size, seq_length)
    attention_mask=attention_mask,
    start_tokens=start_indices,    # Token position where target starts
    end_tokens=end_indices         # Token position where target ends
)

# outputs contains:
# - tag: (batch_size, num_tags)
# - time: (batch_size, num_times)
# - scale: (batch_size, num_scales)
# - negative: (batch_size, 2)
# - gates: {task_name: (batch_size, 1), ...}
```

### 2. `preprocess.py` - Data Processing

**Functions:**
- `convert_span_to_number()`: Convert text spans to numeric values
- `process_batch()`: Tokenize and align target spans with character offsets
- `process_data()`: Parallel batch processing with ThreadPoolExecutor

**Classes:**
- `MultiTaskIterableDataset`: Reads JSONL files with automatic worker sharding
- `BufferedShuffleDataset`: In-memory shuffling for better batch diversity
- `ProcessedIterableDataset`: Loads pre-processed tensor data

**Input Format (JSONL):**
```json
{
  "job_id": "unique_identifier",
  "context": {
    "context_t": "table text",
    "context_p": "paragraph text",
    "context_n": "note text"
  },
  "document": {
    "document_type": "10-K",
    "period_end_date": "2023-12-31",
    "fiscal_year": 2023,
    "period_focus": "FY",
    "document_link": "url"
  },
  "targets": [
    {
      "seq_id": 0,
      "text": "1234567",
      "start_pos": 100,
      "end_pos": 107,
      "attribute": ["tag", "time", "scale", "negative"]
    }
  ],
  "golds": [
    {
      "value": ["us-gaap:Revenues", "instant; current", "6", "0"]
    }
  ]
}
```

**Output Format:**
```python
{
    "input_ids": torch.Tensor,           # Tokenized input
    "attention_mask": torch.Tensor,      # Attention mask
    "start_token": int,                  # Target start token index
    "end_token": int,                    # Target end token index
    "tag": int,                          # Label indices (-100 for missing)
    "time": int,
    "scale": int,
    "negative": int,
    "value": float,                      # Numeric value of span
}
```

### 3. `train.py` - Training Orchestration

**Loss Functions:**
- `CB_CE_Loss`: Class-Balanced Cross-Entropy (handles class imbalance)
- `FocalLoss`: Focal Loss (focuses on hard negatives)

**Key Functions:**
- `set_seed()`: Ensure reproducibility
- `compute_loss()`: Multi-task loss with gate regularization
- `validate_model()`: Validation loop
- `save_checkpoint()`: Save/load training state
- `save_model_config()`: Save model configuration for reproducibility

**Loss Formulation:**

For each task:
```
L_final = (1 - g_task) * L_raw + λ_task * g_task²
```

Where:
- `g_task` ∈ [0, 1] is the learnable gate (noise probability)
- `L_raw` is the task-specific loss
- `λ_task` controls regularization strength

**Example:**
```python
from train import compute_loss, save_model_config

# Inside training loop
outputs = model(input_ids, attention_mask, start_tokens, end_tokens)

targets = {
    "tag": batch["tag"].to(device),
    "time": batch["time"].to(device),
    "scale": batch["scale"].to(device),
    "negative": batch["negative"].to(device)
}

task_weights = {"tag": 10.0, "time": 1.0, "scale": 0.2, "negative": 5.0}
gate_reg_weights = {"tag": 1.0, "time": 1.0, "scale": 1.0, "negative": 1.0}

loss, losses_dict, hits_dict = compute_loss(
    outputs, targets, values,
    task_weights=task_weights,
    gate_reg_weights=gate_reg_weights,
    hits_k=True
)

loss.backward()
```

### 4. `evaluate.py` - Evaluation & Inference

**Metrics:**
- Classification: Precision, Recall, F1, Confusion Matrix
- Ranking: Hits@K
- Regression: MSE, RMSE, MAE, R²

**Main Function:**
- `evaluate_model()`: Full evaluation with error analysis

**Example:**
```python
from evaluate import evaluate_model, load_model_config

# Load config
config = load_model_config("model_weight/secbert/model_config.json")
id2tag = {int(k): v for k, v in config["id2tag"].items()}

# Run evaluation
results = evaluate_model(
    model=model,
    test_loader=test_loader,
    device="cuda",
    id2tag=id2tag,
    id2time=id2time,
    id2scale=id2scale,
    output_dir="results/",
    run_name="secbert-0426-173"
)

# Access results
print(f"Tag F1: {results['tag']['macro_f1']:.4f}")
print(f"Time F1: {results['time']['macro_f1']:.4f}")
```

## Training

### Quick Start

```bash
python train_example.py \
    --train-files processed_data_task1_smaller/train_400k.jsonl \
    --valid-files processed_data_task1_smaller/valid_50k.jsonl \
    --batch-size 32 \
    --epochs 30 \
    --run-name secbert-0426-173
```

### Full Options

```bash
python train_example.py \
    --train-files train.jsonl \
    --valid-files valid.jsonl \
    --model-dir model_weight/secbert \
    --checkpoint-dir check_point/secbert \
    --run-name my-run \
    --batch-size 32 \
    --epochs 30 \
    --patience 3 \
    --eval-steps 10000 \
    --bert-lr 1e-5 \
    --tag-head-lr 5e-4 \
    --time-head-lr 1e-5 \
    --scale-head-lr 3e-5 \
    --negative-head-lr 2e-5 \
    --num-workers 4
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch-size` | 32 | Batch size for training |
| `epochs` | 30 | Number of training epochs |
| `bert-lr` | 1e-5 | Learning rate for BERT |
| `tag-head-lr` | 5e-4 | Learning rate for tag head |
| `eval-steps` | 10000 | Steps between validation runs |
| `patience` | 3 | Early stopping patience |

### Output Files

After training, the following files are created:

```
model_weight/secbert/
├── secbert-0426-173_step820000.pt        # Best model weights
└── secbert-0426-173_config.json          # Model configuration

check_point/secbert/
├── 0426_173_step10000.pth                # Training checkpoints
└── ...
```

## Evaluation

### Quick Start

```bash
python evaluate_example.py \
    --model-path model_weight/secbert/secbert-0426-173_step820000.pt \
    --config-path model_weight/secbert/secbert-0426-173_config.json \
    --test-files processed_data_task1_smaller/test_50k.jsonl \
    --output-dir result/secbert
```

### Output Files

```
result/secbert/
├── secbert-0426-173_results.json         # Overall metrics
├── secbert-0426-173_errors_tag.csv       # Tag misclassifications with gate values
├── secbert-0426-173_errors_time.csv      # Time errors
├── secbert-0426-173_errors_scale.csv     # Scale errors
└── secbert-0426-173_errors_negative.csv  # Negative detection errors
```

## Advanced Usage

### Custom Loss Functions

```python
from train import CB_CE_Loss, FocalLoss

# Class-balanced loss (for imbalanced classes)
num_samples_per_class = [100, 50, 30, 20]
tag_loss = CB_CE_Loss(num_samples_per_class, beta=0.99)

# Focal loss (for hard negatives)
neg_loss = FocalLoss(alpha=0.25, gamma=3.0, reduction="none")

# In compute_loss, customize task_weights
task_weights = {
    "tag": 10.0,      # High weight for important task
    "time": 1.0,
    "scale": 0.2,     # Low weight for easier task
    "negative": 5.0
}

# Gate regularization weights (λ_task in formula)
gate_reg_weights = {
    "tag": 1.0,       # More regularization = harder to gate
    "time": 0.5,      # Less regularization = easier to ignore noisy samples
    "scale": 1.0,
    "negative": 1.0
}
```

### Loading Checkpoints

```python
from train import load_checkpoint

checkpoint_path = "check_point/secbert/0426_173_step200000.pth"
epoch, step = load_checkpoint(model, optimizer, scheduler, checkpoint_path, device)
print(f"Resumed from epoch {epoch}, step {step}")
```

### Custom Data Loading

```python
from preprocess import MultiTaskIterableDataset, BufferedShuffleDataset
from torch.utils.data import DataLoader

# Load raw JSONL files with on-the-fly processing
dataset = MultiTaskIterableDataset(
    files=["train.jsonl"],
    target_attrs=["tag", "time", "scale", "negative"],
    tokenizer=tokenizer,
    tag2id=tag2id,
    time2id=time2id,
    scale2id=scale2id,
    batch_size=32,
    num_workers=4
)

# Add shuffling
dataset = BufferedShuffleDataset(dataset, buffer_size=8000)

# Create DataLoader
loader = DataLoader(dataset, batch_size=32, collate_fn=my_collate_fn)
```

## Performance Metrics

### Task-Specific Performance

The model typically achieves:

| Task | Metric | Performance |
|------|--------|-------------|
| Tag | Macro F1 | ~0.75-0.85 |
| Time | Macro F1 | ~0.80-0.90 |
| Scale | Macro F1 | ~0.85-0.95 |
| Negative | Macro F1 | ~0.90-0.95 |

### Gate Analysis

- **Mean Gate Value**: ~0.3-0.5 indicates healthy gate learning
- **Gate Std Dev**: Higher values indicate task-specific importance variations
- **Error Analysis**: Gates tend to be higher for misclassified samples

## Troubleshooting

### Out of Memory

- Reduce `batch-size`
- Use gradient accumulation
- Consider using mixed precision training

### Training Loss Not Decreasing

- Check learning rates (especially task-specific head learning rates)
- Verify task weights are appropriate
- Ensure data loading is correct (no corrupted samples)

### Model Diverging

- Reduce learning rates, especially for task heads
- Increase gate regularization weights
- Add gradient clipping (already in train_example.py)

## Configuration Management

The model config is automatically saved during training:

```python
save_model_config(
    model=model,
    tag2id=tag2id,
    time2id=time2id,
    scale2id=scale2id,
    num_tags=200,
    num_times=9,
    num_scales=25,
    output_dir="model_weight/secbert",
    run_name="secbert-0426-173"
)
```

This creates `secbert-0426-173_config.json` containing:
- All label-to-index mappings
- Model architecture parameters
- Training constants (missing value sentinels)

Always load this config during evaluation to ensure consistency:

```python
config = load_model_config("model_weight/secbert/secbert-0426-173_config.json")
```

## Dependencies

```
torch>=1.9.0
transformers>=4.0.0
numpy
pandas
scikit-learn
tqdm
word2number
```

## Installation

```bash
pip install torch transformers numpy pandas scikit-learn tqdm word2number
```

## References

- SecBERT: [nlpaueb/sec-bert-base](https://huggingface.co/nlpaueb/sec-bert-base)
- Class-Balanced Loss: https://ieeexplore.ieee.org/abstract/document/8953804
- Focal Loss: https://arxiv.org/abs/1708.02002
- BERT: https://arxiv.org/abs/1810.04805

## License

[Add appropriate license]

## Citation

If you use this framework, please cite:

```bibtex
@software{secbert_framework,
  title={SecBERT Multi-Task Learning Framework},
  year={2024},
  url={https://github.com/...}
}
```

## Authors

[Your names/organization]
