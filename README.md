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
runs = runs.rename(columns={"tags.mlflow.runName": "run_name"})
print(runs[["run_name", "metrics.combined"]].dropna())
```

`search_runs` does not return a `run_name` column directly; the display name lives in `tags.mlflow.runName`, hence the rename. If nothing prints, check `mlflow.get_tracking_uri()` — the default local file store resolves relative to the current working directory, so an unset `mlflow.tracking_uri` in the config combined with a changed cwd (e.g. after a Colab runtime restart) will silently search an empty store. Pin `mlflow.tracking_uri` in `configs/post_training.yaml` to an absolute path (e.g. `sqlite:////absolute/path/to/mlflow.db`) to avoid this.

## MLflow and evaluation

Set `MLFLOW_TRACKING_URI=databricks` for Databricks tracking; otherwise runs use local `./mlruns`. The offline harness reports numeric-tolerance exact match, span match and a combined score for base, SFT and DPO outputs.

## Results

`compare_models.py` on 50 held-out TAT-QA prompts, greedy decoding, seed 42:

| Stage | numeric_em | span_match | combined |
| ----- | ---------: | ---------: | -------: |
| Base  |       0.20 |       0.50 |     0.35 |
| SFT   |       0.50 |       0.50 |     0.50 |
| DPO   |       0.70 |       0.70 |     0.70 |

SFT's gain is concentrated in `numeric_em` (0.20 → 0.50): supervised fine-tuning teaches the model to extract and format the numeric answer TAT-QA expects, but `span_match` doesn't move on its own. DPO lifts both metrics together (`span_match` 0.50 → 0.70, `numeric_em` 0.50 → 0.70), consistent with preference optimisation rewarding concise, context-grounded answers rather than one specific answer shape.

50 examples is small enough that these numbers should be read as a directional signal, not a tight estimate — see below.

## What I learned / what I'd do differently

- **Eval set size is the weakest link, not the training recipe.** 50 examples is enough to see a clear base → SFT → DPO ordering but too few to trust the exact gap between stages. Next iteration: score the full validation split (or several hundred examples) and report confidence intervals rather than point estimates.
- **`span_match` staying flat after SFT was a useful signal, not a bug.** It showed SFT was optimising answer *formatting* rather than answer *correctness* on non-numeric questions, which is exactly the kind of gap DPO is meant to close — the metric split validated the two-stage design instead of just confirming "training helped."
- **Numeric-tolerance EM and span match don't catch fluent-but-wrong reasoning.** Both are surface-form metrics; a model can get the right span for the wrong reason. An LLM-judge or small human-graded sample would be the next addition to confirm the gains reflect real reasoning quality, not metric gaming.
- **Sequential model loading (`compare_models.py` loads/scores/frees one model at a time) was the right call for single-GPU Colab, but it makes the comparison slow to iterate on.** With more VRAM I'd load all three concurrently, or cache generations so metric changes don't require re-running inference.

## Licence

MIT. TAT-QA is distributed under its own CC-BY-4.0 terms; review dataset terms before redistribution.
