# 📑 SecBERT Refactored Project - Complete Index

## 🎯 Project Summary

Your SecBERT Multi-Task Learning model has been successfully refactored from scattered Jupyter notebooks into a **production-ready Python framework**.

**Total Code:** 3,000+ lines of well-documented, modular Python code

## 📂 File Structure

```
model_code/
│
├── 📦 CORE MODULES (Framework)
│   ├── __init__.py              (50 lines)   - Package initialization and exports
│   ├── model.py                 (250 lines)  - Neural network architecture
│   ├── preprocess.py            (500 lines)  - Data loading & processing
│   ├── train.py                 (450 lines)  - Training utilities & losses
│   └── evaluate.py              (450 lines)  - Evaluation & metrics
│
├── 🚀 EXAMPLE SCRIPTS (Ready to use)
│   ├── train_example.py         (380 lines)  - Complete training script
│   └── evaluate_example.py      (290 lines)  - Complete evaluation script
│
├── ⚙️ CONFIGURATION
│   └── config_template.py       (300 lines)  - Configuration template with docs
│
└── 📚 DOCUMENTATION
    ├── QUICK_START.md           (250 lines)  - 5-minute getting started guide
    ├── README_REFACTORED.md     (600 lines)  - Complete API documentation
    ├── REFACTORING_SUMMARY.md   (400 lines)  - What changed and why
    └── INDEX.md                 (This file)  - Complete project index
```

## 🔍 Quick Navigation

### I want to... → Read this

| Goal | File | Section |
|------|------|---------|
| **Get started in 5 minutes** | `QUICK_START.md` | Quick Start (5 minutes) |
| **Understand the architecture** | `README_REFACTORED.md` | Overview + Module Documentation |
| **See what changed** | `REFACTORING_SUMMARY.md` | What Was Done |
| **Train a model** | `train_example.py` | Main function + run `python train_example.py --help` |
| **Evaluate a model** | `evaluate_example.py` | Main function + run `python evaluate_example.py --help` |
| **Customize settings** | `config_template.py` | Copy to `config.py` and edit |
| **Use modules in code** | `__init__.py` | See what's exported |
| **Deep dive into implementation** | Individual module files | Read docstrings |

## 📖 Module Reference

### 1️⃣ `model.py` - Architecture

**Classes:**
- `GateHead(hidden_size, dropout_prob)` - Task-specific gating
- `MultiTaskModel(bert_model_name, num_tags, num_times, num_scales)` - Main model

**Key Methods:**
- `model.forward(input_ids, attention_mask, start_tokens, end_tokens)` → outputs dict

**Usage:**
```python
from model import MultiTaskModel
model = MultiTaskModel("nlpaueb/sec-bert-base", 200, 9, 25)
outputs = model(input_ids, attention_mask, start_tokens, end_tokens)
```

### 2️⃣ `preprocess.py` - Data Pipeline

**Functions:**
- `convert_span_to_number(span)` → float
- `process_batch(batch, target_attrs, tokenizer, ...)` → list[dict]
- `process_data(data, target_attrs, tokenizer, ...)` → list[dict]

**Classes:**
- `MultiTaskIterableDataset` - Read JSONL with worker sharding
- `BufferedShuffleDataset` - In-memory shuffling
- `ProcessedIterableDataset` - Load pre-processed tensors

**Usage:**
```python
from preprocess import MultiTaskIterableDataset
dataset = MultiTaskIterableDataset(
    files=["train.jsonl"],
    target_attrs=["tag", "time", "scale", "negative"],
    tokenizer=tokenizer,
    tag2id=tag2id, time2id=time2id, scale2id=scale2id
)
```

### 3️⃣ `train.py` - Training Infrastructure

**Functions:**
- `set_seed(seed)` - Reproducibility
- `hits_at_k(predictions, targets, k)` → float
- `compute_loss(outputs, targets, values, task_weights, gate_reg_weights, hits_k)` → (loss, losses_dict)
- `validate_model(model, val_loader, task_weights, gate_reg_weights, device)` → (loss, metrics)
- `save_checkpoint(model, optimizer, scheduler, epoch, step, save_path)`
- `load_checkpoint(model, optimizer, scheduler, save_path, device)` → (epoch, step)
- `save_model_config(model, tag2id, time2id, scale2id, num_tags, num_times, num_scales, output_dir, run_name)`

**Classes:**
- `CB_CE_Loss(num_samples, beta, ignore_index)` - Class-balanced loss
- `FocalLoss(alpha, gamma, reduction, ignore_index)` - Focal loss

**Usage:**
```python
from train import CB_CE_Loss, compute_loss, save_model_config

tag_loss = CB_CE_Loss(num_tag_samples, beta=0.99)
loss, losses = compute_loss(outputs, targets, values, task_weights, gate_reg_weights)
save_model_config(model, tag2id, time2id, scale2id, 200, 9, 25, "model_weight", "my-run")
```

### 4️⃣ `evaluate.py` - Evaluation & Metrics

**Functions:**
- `hits_at_k(predictions, targets, k)` → float
- `evaluate_classification(predictions, targets, class_names, id2label)` → dict
- `evaluate_regression(predictions, targets, metric_name)` → dict
- `evaluate_model(model, test_loader, device, id2tag, id2time, id2scale, output_dir, run_name, ...)` → results_dict
- `load_model_config(config_path)` → config_dict
- `get_device()` → str

**Usage:**
```python
from evaluate import evaluate_model, load_model_config

config = load_model_config("model_config.json")
results = evaluate_model(model, test_loader, "cuda", id2tag, id2time, id2scale, "results/", "run1")
```

## 🚀 Common Tasks

### Training a New Model

```bash
python train_example.py \
    --train-files train.jsonl \
    --valid-files valid.jsonl \
    --batch-size 32 \
    --epochs 30 \
    --run-name my-model
```

**Output:**
- ✅ `model_weight/secbert/my-model_step*.pt` (weights)
- ✅ `model_weight/secbert/my-model_config.json` (config)
- ✅ `check_point/secbert/0426_*.pth` (checkpoints)

### Evaluating a Model

```bash
python evaluate_example.py \
    --model-path model_weight/secbert/my-model_step820000.pt \
    --config-path model_weight/secbert/my-model_config.json \
    --test-files test.jsonl \
    --output-dir results/
```

**Output:**
- ✅ `results/my-model_results.json` (metrics)
- ✅ `results/my-model_errors_*.csv` (errors)

### Using in Your Code

```python
# Import modules
from model import MultiTaskModel
from preprocess import MultiTaskIterableDataset
from train import compute_loss, save_model_config
from evaluate import evaluate_model

# Create model
model = MultiTaskModel("nlpaueb/sec-bert-base", 200, 9, 25)

# Create data
dataset = MultiTaskIterableDataset(...)

# Compute loss
outputs = model(input_ids, attention_mask, start_tokens, end_tokens)
loss, losses = compute_loss(outputs, targets, values, task_weights)

# Evaluate
results = evaluate_model(model, test_loader, device, id2tag, id2time, id2scale)
```

## 🎓 Learning Path

**Beginner:**
1. Read `QUICK_START.md`
2. Run training with default settings
3. Review generated results

**Intermediate:**
1. Read `README_REFACTORED.md`
2. Customize `config_template.py`
3. Experiment with task weights and learning rates

**Advanced:**
1. Read `REFACTORING_SUMMARY.md`
2. Modify `model.py` to add custom heads
3. Implement custom loss functions in `train.py`
4. Add new evaluation metrics in `evaluate.py`

## 📊 Model Architecture

```
Input: Text + Target Span Indices
    ↓
[BERT Encoder]
    ↓
[Target Span Averaging]
    ↓
    ├── [Tag Head] ──────→ Tag Logits (batch_size, num_tags)
    ├── [Time Head] ─────→ Time Logits (batch_size, num_times)
    ├── [Scale Head] ────→ Scale Logits (batch_size, num_scales)
    ├── [Negative Head] ─→ Negative Logits (batch_size, 2)
    └── [Gate Head] ─────→ Per-task Gates (batch_size, 4)
```

## 🔄 Loss Computation

```
For each task:
    L_final = (1 - g_task) * L_raw + λ_task * g_task²

Total Loss = sum(task_weight[task] * L_final[task])
```

## 📁 Data Format

### Input (JSONL)

```json
{
  "job_id": "string",
  "targets": [{"text": "123", "start_pos": 0, "end_pos": 3}],
  "golds": [{"value": ["tag_name", "time_class", "scale", "negative"]}],
  "context": {"context_t": "...", "context_p": "...", "context_n": "..."},
  "document": {"document_type": "10-K", "period_end_date": "2023-12-31", ...}
}
```

### Output (Training)

```python
{
    "input_ids": torch.Tensor(512),
    "attention_mask": torch.Tensor(512),
    "start_token": int,
    "end_token": int,
    "tag": int,           # -100 if missing
    "time": int,          # -100 if missing
    "scale": int,         # -100 if missing
    "negative": int,      # -100 if missing
    "value": float        # Numeric value of span
}
```

## ✅ Verification Checklist

- ✅ All modules import correctly
- ✅ Model loads pre-trained BERT
- ✅ Data loading handles JSONL files
- ✅ Loss functions support batch operations
- ✅ Evaluation generates metrics
- ✅ Configuration saves/loads properly
- ✅ Checkpoints save/load without errors

## 🔧 Customization Points

### Add New Task

1. Add head in `model.py`
2. Add loss in `train.py`
3. Add evaluation in `evaluate.py`
4. Update `config_template.py`

### Change BERT Model

Edit `model.py`:
```python
self.bert = BertModel.from_pretrained("different-bert-model")
```

### Add New Loss Function

Add to `train.py`:
```python
class MyLoss(nn.Module):
    def forward(self, outputs, targets):
        # Your loss computation
        return loss_vector  # shape: (batch_size,)
```

### Change Learning Rate

Edit `train_example.py` or command line:
```bash
python train_example.py --bert-lr 2e-5 --tag-head-lr 1e-3
```

## 📚 Documentation Files

| File | Purpose | Length |
|------|---------|--------|
| `QUICK_START.md` | Get running in 5 minutes | ~250 lines |
| `README_REFACTORED.md` | Complete API reference | ~600 lines |
| `REFACTORING_SUMMARY.md` | Design decisions & changes | ~400 lines |
| Module docstrings | Implementation details | In each .py file |

## 🐛 Debugging

### Check imports
```python
python -c "from model import MultiTaskModel; print('✅ OK')"
```

### Check data loading
```bash
python train_example.py --train-files test.jsonl --batch-size 2
```

### Check GPU
```python
import torch
print(f"GPU: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.current_device()}")
```

### Monitor training
- Check `wandb.ai` for live metrics (if enabled)
- Check error CSVs for pattern analysis

## 🎁 What You Get

✅ **Production-Ready Code**
- Well-organized, documented modules
- Proper error handling
- Configuration management

✅ **Complete Examples**
- Training script with all options
- Evaluation script with metrics
- Configuration template

✅ **Comprehensive Documentation**
- Module API reference
- Architecture diagrams
- Usage examples
- Troubleshooting guide

✅ **Reproducibility**
- Automatic config saving
- Checkpoint management
- Seed control

✅ **Extensibility**
- Easy to add custom heads
- Pluggable loss functions
- Modular design

## 🚀 Next Steps

1. **Run Quick Start**: `QUICK_START.md`
2. **Configure**: Copy `config_template.py` → `config.py`
3. **Train**: `python train_example.py ...`
4. **Evaluate**: `python evaluate_example.py ...`
5. **Iterate**: Adjust hyperparameters and repeat

## 📞 Support

For detailed information:
- **Getting Started**: See `QUICK_START.md`
- **API Reference**: See `README_REFACTORED.md`
- **Implementation Details**: See `REFACTORING_SUMMARY.md`
- **Code Help**: Read docstrings in module files

---

**Congratulations!** Your SecBERT codebase is now **production-ready** and **fully modular**! 🎉

For immediate next steps, start with `QUICK_START.md` ➡️
