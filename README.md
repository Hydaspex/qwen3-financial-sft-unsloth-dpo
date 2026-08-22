# qwen3-financial-sft-unsloth-dpo

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hydaspex/qwen3-financial-sft-unsloth-dpo/blob/main/notebooks/colab_pipeline.ipynb)

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
configs/post_training.yaml         # model, LoRA, SFT, DPO and evaluation settings
src/finpost/                       # validated configuration, data, metrics, generation and checkpoints
scripts/                           # preparation, SFT, DPO, comparison and evaluation
notebooks/colab_pipeline.ipynb     # one-click Colab smoke run (small config)
notebooks/kaggle_full_run.ipynb    # full pipeline as an unattended Kaggle commit
tests/                             # CPU-only unit tests
.github/workflows/ci.yml           # Ruff and pytest
```

The Colab notebook runs a reduced smoke configuration interactively — fast, but not the numbers below. The Kaggle notebook runs the full config as a background `Save & Run All` commit (12h session cap, ~30 free GPU-hours/week), independent of keeping a browser tab open, and is what actually produced the results reported here.

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

`compare_models.py` on the full 662-example TAT-QA validation split, seed 42, `batch_size 8`:

| Stage | accuracy | 95% CI |
| ----- | -------: | -----: |
| Base  |   0.1949 | [0.166, 0.225] |
| SFT   |   0.1239 | [0.101, 0.150] |
| DPO   |   0.1405 | [0.116, 0.168] |

Taken at face value, this looks like a regression: base outperforms both fine-tuned stages. That reading does not survive a breakdown by answer type, however, which tells a substantially different story:

| Answer type | n | Base | SFT | DPO |
| --- | --: | --: | --: | --: |
| arithmetic | 281 | 0.0036 | 0.0961 | 0.0996 |
| span | 293 | 0.3515 | 0.1092 | 0.1502 |
| multi-span | 68 | 0.3676 | 0.2353 | 0.2794 |
| count | 20 | 0.00 | 0.35 | 0.10 |

On `arithmetic` — the largest category, and the direct payoff of this session's completion-only-loss fix — fine-tuning takes the base model from near-total failure (0.36%) to roughly 10%, a gain of some 27x. That gain comes at a real cost on `span` and `multi_span`, though: base's verbose, unconstrained generation happens to contain the correct extracted text somewhere in its output more often than either fine-tuned model's concise answers do, and since `span` is nearly as large a category as `arithmetic` (293 examples against 281), the loss there is large enough to outweigh the arithmetic gain in the aggregate score. In other words, the aggregate number is not wrong so much as it is averaging together a large win and a real loss and reporting the difference as if nothing had happened.

This is precisely the failure mode the type-aware scoring fix was meant to catch, and it worked: without the per-type breakdown, this tradeoff would have been invisible, buried inside a single blended number.

The span regression was worth investigating rather than leaving unexplained, since it involved genuinely degenerate output — repetition, empty strings, and one particular hallucinated token (`Intialized`) recurring verbatim across otherwise unrelated examples. Three concrete, checkable causes were ruled out against the real data: training-data truncation corrupting span targets (99.7% of span answers are actually present in the truncated training context, so the targets themselves are sound), prompt-length truncation at evaluation time, and generation-budget truncation (the degenerate outputs are short, not cut off mid-answer at the `max_new_tokens=64` ceiling). A data leak was ruled out too — the hallucinated token appears nowhere in the raw TAT-QA data or the constructed DPO preference pairs. What remains, and what best fits the evidence, is a training-sufficiency explanation: producing a long, exactly-correct extracted span is a harder generation skill than emitting a single number, and a LoRA-adapted model evidently has not converged on it as reliably as it has on short numeric answers.

## What I learned / what I'd do differently

- **The aggregate score hid a real tradeoff, not a null result.** Base beating both fine-tuned stages looked at first like fine-tuning had failed outright — but the per-type breakdown told a different story: a large arithmetic gain offset by a genuine span-extraction regression, two effects a single blended number cannot tell apart from "nothing changed." Scoring by answer type should have been the default from the start.
- **The span regression was worth diagnosing by elimination rather than by guessing.** Training-data truncation, prompt truncation, generation-budget truncation, and a data leak were each checked directly against the real data and ruled out in turn, leaving training-sufficiency as the credible explanation rather than the convenient one.
- **Spot-checking real predictions is what made that diagnosis possible.** The metrics alone would never have shown that SFT/DPO's wrong span answers were degenerate — repetition, empty strings, a recurring hallucinated token — rather than simply incorrect.
- **The full 662-example split with bootstrap CIs has replaced the earlier 50-example directional read**, tight enough now to support the arithmetic-gain claim with real confidence.
- **The natural next step is more span-focused training, not further diagnosis** — more epochs, more span-type examples, or length-aware curriculum ordering, now that this run's data has been checked as thoroughly as is useful.

## Licence

MIT. TAT-QA is distributed under its own CC-BY-4.0 terms; review dataset terms before redistribution.
