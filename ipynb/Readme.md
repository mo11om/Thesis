📖 SecBERT User Guide & CookbookThis guide complements the technical documentation by providing practical recipes for common tasks.1. 🛠️ Configuration Deep DiveThe framework relies on config.py for dataset paths and global constants. You must create this file before running anything.Step 1: Create ConfigCopy the template to create your active configuration file:cp config_template.py config.py
Step 2: Essential Edits in config.pyOpen config.py and modify these specific sections:File Paths (CRITICAL)# UPDATE THESE PATHS to point to your actual data locations
train_files = ["/path/to/your/train.jsonl"]
valid_files = ["/path/to/your/valid.jsonl"]
test_files  = ["/path/to/your/test.jsonl"]
Label CountsYou must update these to match your specific dataset. If you have 150 unique tags, num_tags must be at least 150 (usually max_id + 1).num_tags = 200    # Example: Update this!
num_times = 9     # Example: Update this!
num_scales = 25   # Example: Update this!
Label MappingsEnsure tag2id, time2id, and scale2id dictionaries contain every possible label in your dataset.tag2id = {
    "us-gaap:Revenues": 0,
    "us-gaap:Assets": 1,
    # ... add all your tags
}
2. 🍲 Training RecipesRun these commands from the terminal where train_example.py is located.Recipe A: The "Standard" RunGood for the first full training session.python train_example.py \
    --train-files data/train.jsonl \
    --valid-files data/valid.jsonl \
    --batch-size 32 \
    --epochs 30 \
    --run-name secbert-v1
Recipe B: Resume Training (After Crash/Stop)If your training stopped at step 50,000, you can resume exactly where you left off.python train_example.py \
    --train-files data/train.jsonl \
    --valid-files data/valid.jsonl \
    --checkpoint-dir check_point/secbert \
    --checkpoint check_point/secbert/secbert-v1_step50000.pth \
    --run-name secbert-v1-resumed
Recipe C: Low Memory (OOM Fix)If you get "CUDA Out of Memory" errors, use a smaller batch size and fewer workers.python train_example.py \
    --train-files data/train.jsonl \
    --batch-size 16 \
    --num-workers 1 \
    --run-name secbert-low-mem
Recipe D: Hyperparameter TuningAdjust learning rates for specific parts of the model. For example, if the BERT backbone is overfitting but the classification heads are underfitting:python train_example.py \
    --bert-lr 5e-6 \       # Lower LR for BERT
    --tag-head-lr 1e-3 \   # Higher LR for Tag Head
    --epochs 50 \
    --run-name secbert-tuned
3. 📊 Evaluation & AnalysisAfter training, you have a model file (e.g., model_weight/secbert/secbert-v1_step820000.pt).Running Evaluationpython evaluate_example.py \
    --model-path model_weight/secbert/secbert-v1_step820000.pt \
    --config-path model_weight/secbert/secbert-v1_config.json \
    --test-files data/test.jsonl \
    --output-dir results/v1_analysis
Note: Always use the _config.json generated during training. It ensures the label IDs match exactly.Interpreting Outputsresults.json (Metrics)Macro F1: The average performance across all classes.Hits@K: Useful for "Tag" prediction. If Hits@3 is 95%, the correct tag is in the top 3 predictions 95% of the time.errors_*.csv (The Gold Mine)This file lists every mistake the model made. Open it in Excel/Pandas.Columns: text, true_label, pred_label, gate_value.Analysis Tip: Sort by gate_value (descending). High gate values (near 1.0) mean the model thought this sample was "noisy" or "confusing."High Gate + Wrong Prediction: The model knew it was confused.Low Gate + Wrong Prediction: The model was confident but wrong (Dangerous error).4. 💻 Python Inference SnippetHow to use the trained model in your own Python script to make predictions.import torch
from transformers import BertTokenizer
from model import MultiTaskModel
from evaluate import load_model_config

# 1. Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = BertTokenizer.from_pretrained("nlpaueb/sec-bert-base")

# 2. Load Configuration & Model
config_path = "model_weight/secbert/secbert-v1_config.json"
model_path = "model_weight/secbert/secbert-v1_step820000.pt"

config = load_model_config(config_path)
model = MultiTaskModel(
    "nlpaueb/sec-bert-base",
    num_tags=config['num_tags'],
    num_times=config['num_times'],
    num_scales=config['num_scales']
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 3. Prepare Input (Simulated Single Example)
text = "Revenues for the quarter were $50 million."
target_span_text = "50 million"
# You need to find the start/end index of the span in the tokenized output
encoded = tokenizer(text, return_tensors="pt")
input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

# Find token indices for "50 million" (Simplified logic)
# In production, use the preprocess.py logic for robust aligning
start_token_idx = 5  # Example index
end_token_idx = 7    # Example index

# 4. Predict
with torch.no_grad():
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        start_tokens=torch.tensor([start_token_idx]).to(device),
        end_tokens=torch.tensor([end_token_idx]).to(device)
    )

# 5. Decode Results
pred_tag_id = outputs['tag'].argmax(dim=1).item()
pred_tag_name = config['id2tag'][str(pred_tag_id)] # JSON keys are strings

print(f"Predicted Tag: {pred_tag_name}")
print(f"Gate Value (Noise Score): {outputs['gates']['tag'].item():.4f}")
5. ❓ Troubleshooting Common IssuesSymptomProbable CauseFixKeyError: 'us-gaap:NewTag'Your training data contains a tag not in tag2id in config.py.Add the missing tag to tag2id in config.py or filter your data.RuntimeError: CUDA error: device-side assert triggeredA label ID in your data is larger than num_tags.Check num_tags in config. It must be > max(label_ids).FileNotFoundErrorThe paths in config.py are relative or incorrect.Use absolute paths (e.g., /home/user/data/...) to be safe.Loss is NaNLearning rate is too high or gradients exploding.Reduce --tag-head-lr or enable gradient clipping (default is 1.0).