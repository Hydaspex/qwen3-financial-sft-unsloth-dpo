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
scripts/                         # preparation, SFT, DPO and evaluation entry points
notebooks/colab_repro.ipynb      # one-click Colab workflow (to be added)
tests/                           # CPU-only unit tests
.github/workflows/ci.yml         # Ruff and pytest
```

## Quick start

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

python scripts/prepare_data.py --config configs/post_training.yaml
python scripts/train_sft.py --config configs/post_training.yaml
python scripts/train_dpo.py --config configs/post_training.yaml \
  --sft-adapter outputs/qwen3-financial-sft
```

For a Colab run, select a GPU runtime, install the project, reduce `max_samples`, and execute the same three stages. The training entry points detect bf16 support and fall back to fp16 where required.

## MLflow and evaluation

Set `MLFLOW_TRACKING_URI=databricks` for Databricks tracking; otherwise runs use local `./mlruns`. The offline harness reports numeric-tolerance exact match, span match and a combined score for base, SFT and DPO outputs.

## Licence

MIT. TAT-QA is distributed under its own CC-BY-4.0 terms; review dataset terms before redistribution.
