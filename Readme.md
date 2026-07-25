# XBRL Multi-Task Tagging — Handover

> Authoritative doc for this repo. `ipynb/Readme.md` is **stale** — it documents `config.py`,
> `train_example.py` and CLI flags that do not exist. Ignore it.

## 1. Environment

Python **3.10.18**, conda env `XBRL` (transformers 4.56.1, numpy 1.26.4, pytorch-cuda 12.4).

```bash
conda env create -f /home/mo1om/code/XBRL/env.yml   # note: one level ABOVE the repo
conda activate XBRL
pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install wandb word2number       # imported by the code, absent from env.yml
wandb login                         # gate_train.py fails without it
```

The `# - pip:` block in `env.yml` (~line 348) is **commented out**, so `conda env create` alone
does **not** install torch. There is no `requirements.txt`, `pyproject.toml`, `Dockerfile`,
`Makefile`, or CI config anywhere — don't go looking.

## 2. Code map

All tracked source is flat at the repo root. **No script takes CLI arguments** (except the
tuning driver); everything is driven by `config.json` and module-level constants.

| File | Role |
|---|---|
| [config.json](config.json) | Single source of truth for a run. `model.{name,date,index}` derives `run_name` and **every** output path |
| [shared_utils.py](shared_utils.py) | The library: preprocessing, dataset classes, `MultiTaskModel`, `GateHead`, losses, all `evaluate_*` |
| [gate_train.py](gate_train.py) | **Main training script** (jupytext export of `secbert_smaller_teacher.ipynb`) |
| [run_inference.py](run_inference.py) | Test-set inference → prediction + error CSVs |
| [analyze_results.ipynb](analyze_results.ipynb) | Those CSVs → accuracy / P-R-F1 / confusion matrices |
| [secbert_smaller_teacher_preprocess.ipynb](secbert_smaller_teacher_preprocess.ipynb) | **Only** preprocessing entry point — notebook, no `.py` twin |
| [hyperparameter_tuning.py](hyperparameter_tuning.py) | Grid sweep over `tuning_spec.json`. Broken: it scrapes val loss from stdout and always logs `inf`. Overwrites `config.json` in place — use `--dry-run` first |
| [sister/](sister) | 5 self-contained baseline notebooks (bert, finbert, secbert, secbertnum, lstm) — no gate, no `shared_utils` |
| `secbert_smaller_teacher_test*.ipynb` | 3 near-duplicate eval notebooks; `_error_gate` is newest and the only one logging **per-task** gates |

### Where things live

Everything below is `.gitignore`d and exists **only on this machine**, under `model_code/`.
Raw data is one level up in `../processed_data_task1_smaller/`.

| Path | Contents |
|---|---|
| **`check_point/<model>/`** | **All trained models, including the baselines** — full training state (`.pth`: model + optimizer + scheduler + step), saved every 10 000 steps. Subdirs: `secbert` (26), `bge` (21), `modernbert` (19), `ModernBERT` (8), `secbert_selfmix` (9), `secbert_coteaching` (8). **Start here when looking for a model.** The one exception: the 5 `sister/` encoder baselines have no `check_point/` dir — the only weights are 4 `.pt` files in `model_weight/secbert/sister/`, and none for bert / finbert / lstm / secbertnum. |
| `model_weight/<model>/` | Best-only `.pt` state_dicts — far sparser than `check_point/`, several dirs empty. Also holds the `<run_name>_config.json` label maps that `run_inference.py` needs, and the gate-λ ablation / variant subdirs under `model_weight/secbert/` (`1111`, `05_10_10_10`, `7_10_10_10`, `N_layer`, `mixup`, `SSR`, `noise`, `sister`, …) |
| `processed_iterable_dataset/` | Tokenized data (train 3.0 GB) + `counter/` class-frequency files |
| `result/`, `error_analysis/`, `gate/` | Prediction CSVs, per-sample error dumps, gate analyses |
| `log/`, `wandb/` | Tuning trial configs, W&B run data |

A `check_point/*.pth` is a full training state, not a bare state_dict — load it with
`shared_utils.load_checkpoint`, not `torch.load` straight into the model.

**Branches.** `gate` is the working branch. Noise-robust baselines (`co_teach`, `selfmix`,
`SSR`, `mixup`, `N_Layer`) and other backbones (`modern`, `bge`) live on **unmerged branches**
of the same name — their checkpoints are already in `check_point/` and `model_weight/`.

## 3. How to run

**`cd` into `model_code/` first** — every path in every script is relative to it.

```bash
cd /home/mo1om/code/XBRL/model_code

# 1. PREPROCESS — Jupyter, run all cells: secbert_smaller_teacher_preprocess.ipynb
#    in : ../processed_data_task1_smaller/{train_400k,valid_50k,test_50k}.jsonl
#         ../processed_data_task1_smaller/counter/tag_count_train_400k.json
#    out: processed_iterable_dataset/{train_400k,valid_50k,test_50k}.jsonl   (512-token)
#         processed_iterable_dataset/counter/secbert_train_small_{tag,time,scale,negative}.json
#    Both outputs already exist — skip unless changing tokenizer or data.
#    The count_classes_in_jsonl cell is REQUIRED: class-balanced losses read those counters.

# 2. TRAIN — edit config.json first (model.name/date/index control all output paths)
python gate_train.py
#    out: check_point/secbert/{date}_{index}_step*.pth  (full state, every eval_step=10000)
#         model_weight/secbert/{date}_{index}_step*.pt  (best only, keeps 2, patience 3)

# 3. INFERENCE — edit constants at run_inference.py:34-52 (RUN_DATE/RUN_INDEX/CHECKPOINT_STEP)
python run_inference.py
#    out: result/secbert_{date}_{index}_result_{tag,time,scale,negative,fact}.csv
#         error_analysis/secbert_{date}_{index}_errors_{attr}.csv   (incl. per-sample gate)

# 4. METRICS — Jupyter: analyze_results.ipynb (set RUN_DATE/RUN_INDEX in cell 2)
```

**Two things will bite you on a fresh run:**

- `gate_train.py` **pins a W&B run id** — `id='wqugd5cx', resume="allow"` at
  [gate_train.py:712-716](gate_train.py:712). Every fresh run silently appends to that stale
  run. Change it before training.
- `run_inference.py` ships pointing at a checkpoint that **does not exist**:
  `CHECKPOINT_STEP = 20001` → `model_weight/secbert/0426_173_step20001.pt`, but only
  `step20000.pt` / `step40000.pt` are on disk. Set it to 40000 or it raises `FileNotFoundError`.

Current `config.json`: batch 64, 30 epochs, LRs `bert 1e-5 / tag 5e-4 / time 1e-5 / scale 3e-5 /
negative 2e-5`, task weights `{tag 10, time 0.3, scale 0.2, negative 10}`, gate λ
`{tag 1, time 10, scale 10, negative 10}`, AdamW + cosine anneal.

Note two tag vocabularies coexist: `gate_train.py:51` and `run_inference.py:308` hardcode
`../processed_data_task1_smaller/counter/tag_count_train_400k.json` (outside the repo), while
`shared_utils.load_counter` uses `processed_iterable_dataset/counter/`. Both are needed;
neither is configurable.

## 4. Baselines on other branches

Each baseline is a **separate branch**, not a flag. Workflow is always
`git checkout <branch>` in `model_code/`, then run. `processed_iterable_dataset/`,
`check_point/`, `model_weight/` etc. are gitignored, so they survive the checkout and are
shared by every branch — which is convenient, and also the main hazard (see below).

### Noise-robust baselines

These **predate the refactor**: the library is `secbert_utils.py` (not `shared_utils.py`) and
there is **no `config.json`** — settings are a dict or constants at the top of each file.

| Branch | Method | Train | Test | Writes to |
|---|---|---|---|---|
| `co_teach` | Co-teaching, two networks cross-select low-loss samples. `noise_rate` 0.1, `num_gradual` 1, batch 32 (config dict at `coteaching_train.py:49-88`) | `python coteaching_train.py` | `secbert_smaller_teacher_test.ipynb` | `secbert_coteaching/`, `0213_1_step*` |
| `selfmix` | SelfMix. Constants at `selfmix.py:34-46` | `python selfmix.py` | `secbert_smaller_test.ipynb` | `secbert_selfmix/`, `0426_173_step*` |
| `SSR` | Sample selection + relabelling (θ_r 0.9, kNN k=50, λ_fc 1.0), config in `train_ssr.py:312-340` | `python train_ssr.py` | `python divide.py` (splits test into a clean subset → `test_50k_clean.jsonl`), or `divid.ipynb` | `secbert/`, `0426_SSR_173_best.pt` |
| `mixup` | mixup α=0.4 — mixing happens in `MultiTaskModel.forward(do_mixup=True, alpha=…)` (`secbert_utils.py:508`) with `compute_loss_with_mixup` (`secbert_utils.py:877`). `reference_utils.py` is just a copy of the utils, no mixup code | `secbert_smaller.ipynb` (notebook only) | `secbert_smaller_teacher_test_mixup.ipynb` | `secbert/` → hand-moved to `model_weight/secbert/mixup/` |
| `N_Layer` | Noise-adaptation layer — `NoiseLayer` at `secbert_utils.py:314`, one per head, applied to the softmax | `secbert_smaller.ipynb` (notebook only) | `secbert_smaller_test.ipynb` | `secbert/` → hand-moved to `model_weight/secbert/N_layer/` |

> ⚠️ **`mixup` and `N_Layer` will overwrite the main run.** Both use `model_name="secbert"`,
> `date="0426"`, `index=173` — identical to the `gate` run — so they write straight over
> `check_point/secbert/0426_173_step*.pth` and `model_weight/secbert/0426_173_step*.pt`. The
> existing `model_weight/secbert/{mixup,N_layer,noise,SSR}/` subdirs were moved there **by hand
> after the fact**. Change `model_name` or `index` before running either. `SSR` is safer only
> because it tags its output `0426_SSR_173_best.pt`.

Three more traps:

- `test_coteaching.py` is a **smoke test of the loss function** (synthetic tensors, asserts,
  "All tests passed") — it does *not* evaluate a trained model. Use the test notebook for that.
- `train_ssr.py` ships with **placeholder class statistics** —
  `tag_list = [f"tag_{i}" for i in range(978)]`, `num_tag_samples = [1] * len(tag_list)`, marked
  "Update with actual loaded counts" — so the class-balanced loss is wrong until you wire in the
  real counters from `processed_iterable_dataset/counter/`.
- In `divide.py:194` the `model.load_state_dict(torch.load(model_save_path))` line is
  **commented out**, so as shipped it selects the "clean" subset using features from an
  *untrained* model. Uncomment it (and check `model_save_path`) before trusting
  `test_50k_clean.jsonl`.

### Alternate backbones

Post-refactor: same `shared_utils.py` + `config.json` design as `gate`, same gate head, just a
different encoder and its own preprocessed data directory.

| Branch | Backbone | Train | Test |
|---|---|---|---|
| `modern` | `answerdotai/ModernBERT-base` | `python gate_train_modern.py` | `python gate_train_modern_inference.py` |
| `bge` | `BAAI/bge-base-en` | `python gate_train_bge.py` | `python gate_train_bge_inference.py` |

Their `config.json` is already set (`ModernBERT` 0513/173 reading
`processed_iterable_dataset_modern/`; `bge` 0601/173 reading `processed_iterable_dataset_bge/`).
Both data dirs already exist — but if you rebuild them, **re-run preprocessing with that
branch's tokenizer**, since token offsets differ per backbone. Set `CHECKPOINT_STEP` near the
top of `main()` in the inference script (currently 50001 for ModernBERT, 90001 for bge) to a
`.pt` that actually exists in `model_weight/{ModernBERT,bge}/`. Note the `bge` branch carries
the `modern` scripts too, and `gate_train_bge_inference.py:21` still defaults its counter dir
to `processed_iterable_dataset_modern/counter` — it overrides this later for the real counters,
but check it if numbers look wrong.