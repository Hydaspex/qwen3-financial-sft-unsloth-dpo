# qwen3-financial-sft-unsloth-dpo

Portfolio-grade financial LLM post-training pipeline: **Unsloth QLoRA supervised fine-tuning (SFT)** followed by **direct preference optimisation (DPO)** on Qwen3-4B, with MLflow tracking, an evaluation harness, and Colab-ready entry points.


## Pipeline

```text
TAT-QA / financial QA
        │
        ▼
chat-format preparation ──► Unsloth QLoRA SFT ──► preference pairs ──► DPO
                                                                  │
                                                                  ▼
                                             numeric/span evaluation + MLflow
```

SFT teaches grounded financial answer formatting. DPO then prefers concise, context-grounded answers over verbose, unsupported or numerically incorrect alternatives.

## Repository layout

```text
configs/post_training.yaml       # model, LoRA, SFT, DPO and evaluation settings
src/finpost/                     # validated configuration, data and metrics
scripts/                         # preparation, SFT, DPO, comparison and evaluation
notebooks/colab_pipeline.ipynb   # one-click Colab GPU workflow
tests/                           # CPU-only unit tests
.github/workflows/ci.yml         # Ruff and pytest
```

## Quick start

The `finpost` package lives in `src/finpost/`. Install it from the repository root:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .          # editable install of finpost
uv pip install -e ".[dev]"   # optional: dev tools (pytest, ruff)

python scripts/prepare_data.py --config configs/post_training.yaml
python scripts/train_sft.py --config configs/post_training.yaml
python scripts/train_dpo.py --config configs/post_training.yaml \
  --sft-adapter outputs/qwen3-financial-sft
```

For a Colab run, select a GPU runtime, clone the repo, run `pip install -q -e ".[dev]"` from the repo root, and execute the same stages. The training scripts include a `sys.path` fallback so `finpost` imports even if the editable install is stale.

## Comparing base, SFT and DPO

After training both adapters, generate and score predictions from all three models on the same held-out validation prompts:

```bash
python scripts/compare_models.py \
  --config configs/post_training.yaml \
  --sft-adapter outputs/qwen3-financial-sft \
  --dpo-adapter outputs/qwen3-financial-dpo \
  --limit 50
```

This prints a table of `numeric_em`, `span_match` and `combined` per stage, logs each stage as a nested MLflow run under `{experiment_name}-comparison`, and writes predictions to `outputs/predictions_{stage}.jsonl`. Iterate over combined scores across experiments with:

```python
import mlflow
runs = mlflow.search_runs(experiment_names=["/Shared/qwen3-financial-sft-unsloth-dpo"])
print(runs[["run_name", "metrics.combined"]].dropna())
```

## MLflow and evaluation

Set `MLFLOW_TRACKING_URI=databricks` for Databricks tracking; otherwise runs use local `./mlruns`. The offline harness reports numeric-tolerance exact match, span match and a combined score for base, SFT and DPO outputs.

## Licence

MIT. TAT-QA is distributed under its own CC-BY-4.0 terms; review dataset terms before redistribution.
