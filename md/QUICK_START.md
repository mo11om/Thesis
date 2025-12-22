# SecBERT Quick Start Guide

## 📦 What You Have

Your SecBERT codebase has been successfully refactored into 8 core modules:

| File | Purpose | Lines |
|------|---------|-------|
| **model.py** | Neural network architecture | 250+ |
| **preprocess.py** | Data loading & tokenization | 500+ |
| **train.py** | Training loop & loss functions | 450+ |
| **evaluate.py** | Evaluation & metrics | 450+ |
| **__init__.py** | Package initialization | 50 |
| **train_example.py** | Complete training script | 380 |
| **evaluate_example.py** | Complete evaluation script | 290 |
| **config_template.py** | Configuration template | 300+ |

## 🚀 Quick Start (5 minutes)

### Step 1: Setup

```bash
# Navigate to your model code directory
cd /home/mo1om/code/XBRL/model_code

# Install dependencies (if not already done)
pip install torch transformers numpy pandas scikit-learn tqdm word2number

# Verify imports
python -c "from model import MultiTaskModel; print('✅ Setup OK')"
```

### Step 2: Configure

```bash
# Copy configuration template
cp config_template.py config.py

# Edit config.py with your data paths
# vim config.py
```

Key settings to update in `config.py`:
- `train_files`, `valid_files`, `test_files`: Your JSONL data paths
- `num_tags`, `num_times`, `num_scales`: Based on your dataset
- `batch_size`, `epochs`: Your training preferences
- Learning rates if needed

### Step 3: Train

```bash
# Train the model
python train_example.py \
    --train-files your_train.jsonl \
    --valid-files your_valid.jsonl \
    --batch-size 32 \
    --epochs 30 \
    --run-name my-model

# Output:
# ✅ model_weight/secbert/my-model_step820000.pt (best weights)
# ✅ model_weight/secbert/my-model_config.json (configuration)
```

### Step 4: Evaluate

```bash
# Evaluate the trained model
python evaluate_example.py \
    --model-path model_weight/secbert/my-model_step820000.pt \
    --config-path model_weight/secbert/my-model_config.json \
    --test-files your_test.jsonl \
    --output-dir results/my-model

# Output:
# ✅ results/my-model/my-model_results.json (metrics)
# ✅ results/my-model/my-model_errors_*.csv (error analysis)
```

### Step 5: Review Results

```bash
# Check overall metrics
cat results/my-model/my-model_results.json | python -m json.tool

# Analyze errors
head -20 results/my-model/my-model_errors_tag.csv

# Summary
echo "Training complete! Check results/ directory."
```

## 📚 Documentation

### For Complete Documentation
👉 See **`README_REFACTORED.md`** for:
- Detailed API reference
- Advanced usage examples
- Troubleshooting guide
- Performance benchmarks

### For Refactoring Details
👉 See **`REFACTORING_SUMMARY.md`** for:
- What was refactored and why
- Design decisions
- Migration guide from notebooks
- Benefits summary

## 🧠 Core Concepts

### The Model Architecture

```
Text Input
    ↓
[BERT Encoder] → Sequence Embeddings
    ↓
[Target Span Extraction] → Span-specific Embeddings
    ↓
    ├→ [Tag Head] → Tag Classification
    ├→ [Time Head] → Time Classification
    ├→ [Scale Head] → Scale Classification
    ├→ [Negative Head] → Negative Classification
    └→ [Gate Head] → Per-task Noise Scores
```

### Gate Mechanism

```
Final Loss = (1 - gate) * Raw Loss + λ * gate²
```

- **gate = 0**: Sample is clean, use full loss
- **gate = 1**: Sample is noisy, mostly ignore
- The model learns which samples are noisy per task!

### Data Format (JSONL)

```json
{
  "job_id": "unique_id",
  "targets": [{"text": "1234567", "start_pos": 100, "end_pos": 107, ...}],
  "golds": [{"value": ["tag_name", "time_class", "scale", "is_negative"]}],
  "context": {"context_t": "...", "context_p": "...", "context_n": "..."},
  "document": {"document_type": "10-K", "period_end_date": "2023-12-31", ...}
}
```

## 🔧 Customization

### Change Learning Rates

Edit `train_example.py`:
```python
parser.add_argument("--bert-lr", type=float, default=1e-5)
parser.add_argument("--tag-head-lr", type=float, default=5e-4)
```

Or pass on command line:
```bash
python train_example.py --bert-lr 2e-5 --tag-head-lr 1e-3
```

### Change Task Weights

In training code:
```python
task_weights = {
    "tag": 10.0,      # Emphasize tag classification
    "time": 1.0,
    "scale": 0.2,     # De-emphasize easier tasks
    "negative": 5.0
}
```

### Change Loss Functions

In `train.py`:
```python
from train import CB_CE_Loss, FocalLoss

# Use class-balanced loss
tag_loss = CB_CE_Loss(num_tag_samples, beta=0.99)

# Use focal loss for hard negatives
neg_loss = FocalLoss(alpha=0.25, gamma=3.0)
```

## 📊 Monitoring Training

### With Weights & Biases

```bash
# Enable W&B logging (default)
python train_example.py --train-files ... 

# View dashboard at wandb.ai
```

### Without W&B

```bash
# Disable W&B
python train_example.py --train-files ... --no-wandb
```

Checkpoints are saved every `eval-steps` (default 10000):
```
check_point/secbert/
├── 0426_173_step10000.pth
├── 0426_173_step20000.pth
└── ...
```

## 🎯 Performance Tips

### For Faster Training
- Increase `--batch-size` (if GPU memory allows)
- Reduce `--eval-steps` for frequent validation
- Use `--num-workers` for parallel data loading

### For Better Accuracy
- Train longer (`--epochs 50`)
- Use smaller learning rates (`--bert-lr 5e-6`)
- Adjust `task_weights` based on task importance
- Collect more training data

### For Lower Memory Usage
- Decrease `--batch-size` to 16 or 8
- Use CPU: `device = "cpu"` in evaluation
- Load data in streaming mode (default)

## 🐛 Common Issues

### Out of Memory
```bash
# Reduce batch size
python train_example.py --batch-size 16
```

### Model not improving
```bash
# Check task weights - are they reasonable?
# Try different learning rates
python train_example.py --bert-lr 5e-6 --tag-head-lr 1e-4
```

### Model diverging
```bash
# Enable gradient clipping (already done in train_example.py)
# Reduce learning rates
# Increase batch size
```

### Configuration mismatch between train & eval
```bash
# Always load config during evaluation!
python evaluate_example.py \
    --config-path model_weight/secbert/my-model_config.json
```

## 📝 Example Workflow

```bash
# 1. Setup
cd /home/mo1om/code/XBRL/model_code
cp config_template.py config.py
vim config.py  # Update your data paths

# 2. Train
python train_example.py \
    --train-files ../processed_data_task1_smaller/train_400k.jsonl \
    --valid-files ../processed_data_task1_smaller/valid_50k.jsonl \
    --batch-size 32 \
    --epochs 30 \
    --run-name secbert-experiment-1

# 3. Evaluate
python evaluate_example.py \
    --model-path model_weight/secbert/secbert-experiment-1_step820000.pt \
    --config-path model_weight/secbert/secbert-experiment-1_config.json \
    --test-files ../processed_data_task1_smaller/test_50k.jsonl \
    --output-dir results/experiment-1

# 4. Analyze
python -c "
import json
with open('results/experiment-1/secbert-experiment-1_results.json') as f:
    results = json.load(f)
    print('Tag F1:', results['tag']['macro_f1'])
    print('Time F1:', results['time']['macro_f1'])
    print('Scale F1:', results['scale']['macro_f1'])
"

# 5. Review errors
head -10 results/experiment-1/secbert-experiment-1_errors_tag.csv
```

## 🔄 Resume Training

```bash
# From checkpoint
python train_example.py \
    --train-files ... \
    --checkpoint check_point/secbert/0426_173_step500000.pth
```

## 🎓 Learning Resources

- **BERT Architecture**: https://arxiv.org/abs/1810.04805
- **Class-Balanced Loss**: https://ieeexplore.ieee.org/abstract/document/8953804
- **Focal Loss**: https://arxiv.org/abs/1708.02002
- **Multi-Task Learning**: https://arxiv.org/abs/1707.02904

## ✅ Validation Checklist

Before training on new data:

- [ ] Data is in JSONL format with correct schema
- [ ] Label mappings (tag2id, time2id, scale2id) are computed
- [ ] Config paths are correct
- [ ] GPU has sufficient memory
- [ ] Dependencies installed (`pip install ...`)
- [ ] Weights & Biases account created (optional)

## 🆘 Getting Help

1. Check **`README_REFACTORED.md`** for detailed docs
2. Review **`REFACTORING_SUMMARY.md`** for design decisions
3. Check example scripts for usage patterns
4. Read docstrings: `python -c "from model import MultiTaskModel; help(MultiTaskModel)"`

## 🎉 Next Steps

1. **Quick Experiment**: Run with default config to test setup
2. **Tune Hyperparameters**: Adjust learning rates and task weights
3. **Experiment with Architectures**: Modify `model.py` if needed
4. **Production Deployment**: Add monitoring and logging

---

**Congratulations!** Your SecBERT codebase is now production-ready! 🚀
