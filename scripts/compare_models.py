"""Compare base, SFT and DPO models on the held-out validation set.

Generates predictions from each model on identical prompts, scores them with
the offline harness, logs metrics (with bootstrap CIs) to MLflow as nested
runs, and prints a comparison table. Models are loaded sequentially to fit
limited VRAM; within each model, prompts are generated in batches.

By default this evaluates the full validation split (--limit 0). Pass a
positive --limit to cap it, e.g. for a quick smoke run.

Usage:
    python scripts/compare_models.py \
        --config configs/post_training.yaml \
        --sft-adapter outputs/qwen3-financial-sft \
        --dpo-adapter outputs/qwen3-financial-dpo \
        --batch-size 8

Note on batched generation: left-padding plus an explicit pad_token_id keeps
batched decoding numerically equivalent to single-example decoding for real
(non-pad) tokens in exact arithmetic. The padding/slicing logic itself
(batch_generate) is unit tested on CPU with a tiny HF model in
tests/test_pipeline.py, so that correctness question doesn't require a GPU to
verify. What CPU testing can't rule out is CUDA/attention kernel selection
varying with batch size on real GPU hardware; if you want to double-check
that too, generate the same ~15 examples with --batch-size 1 and your
intended batch size on Unsloth/CUDA and diff predictions/scores.
"""

import argparse
import gc
import json
import sys
from pathlib import Path

import mlflow
import torch
import unsloth  # noqa: F401
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finpost.config import load_config
from finpost.evaluate import bootstrap_ci, per_example_results, score
from finpost.generate import batch_generate


def jsonl_records(path: str) -> list[dict]:
    """Read newline-delimited JSON records."""
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_model_for_inference(model_path: str, max_seq_length: int):
    """Load a model via Unsloth and set it up for inference.

    Thin and Unsloth-specific by design, so it stays GPU-only without pulling
    the testable batching logic in finpost.generate.batch_generate along with
    it — see that function's docstring for why the split matters.
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_predictions(
    model_path: str,
    records: list[dict],
    max_seq_length: int,
    max_new_tokens: int = 64,
    batch_size: int = 8,
) -> list[str]:
    """Load a model, batch-generate answers for validation prompts, then release VRAM."""
    model, tokenizer = load_model_for_inference(model_path, max_seq_length)

    prompts = [
        tokenizer.apply_chat_template(
            record["messages"][:-1],
            tokenize=False,
            add_generation_prompt=True,
        )
        for record in records
    ]

    predictions = batch_generate(
        model, tokenizer, prompts, max_seq_length, max_new_tokens, batch_size
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--dpo-adapter", required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap the number of validation examples (0 = use the full split).",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default="outputs/comparison_results.json")
    args = parser.parse_args()

    config = load_config(args.config)
    records = jsonl_records(str(config.data.validation_path))
    if args.limit > 0:
        records = records[: args.limit]
    golds = [record["messages"][-1]["content"] for record in records]

    stages = {
        "base": config.model.name_or_path,
        "sft": args.sft_adapter,
        "dpo": args.dpo_adapter,
    }

    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment)

    results = {}
    with mlflow.start_run(run_name=f"{config.experiment_name}-comparison"):
        mlflow.log_params({
            "n_eval": len(records),
            "base_model": config.model.name_or_path,
            "sft_adapter": args.sft_adapter,
            "dpo_adapter": args.dpo_adapter,
        })
        for stage, model_path in stages.items():
            print(f"\n=== Generating with {stage}: {model_path} ===")
            predictions = generate_predictions(
                model_path,
                records,
                config.model.max_seq_length,
                batch_size=args.batch_size,
            )
            metrics = score(predictions, golds)
            example_results = per_example_results(predictions, golds)
            _, numeric_lo, numeric_hi = bootstrap_ci([r["numeric_em"] for r in example_results])
            _, span_lo, span_hi = bootstrap_ci([r["span_match"] for r in example_results])
            ci = {
                "numeric_em_ci_low": numeric_lo,
                "numeric_em_ci_high": numeric_hi,
                "span_match_ci_low": span_lo,
                "span_match_ci_high": span_hi,
            }
            results[stage] = {"predictions": predictions, "metrics": metrics, "ci": ci}
            with mlflow.start_run(run_name=stage, nested=True):
                mlflow.log_param("model_path", model_path)
                mlflow.log_param("n_eval", len(records))
                mlflow.log_metrics(metrics)
                mlflow.log_metrics(ci)
                pred_path = Path(f"outputs/predictions_{stage}.jsonl")
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                pred_path.write_text(
                    "\n".join(
                        json.dumps({"index": i, **r}) for i, r in enumerate(example_results)
                    ),
                    encoding="utf-8",
                )
                mlflow.log_artifact(str(pred_path), artifact_path="predictions")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Comparison ===")
    print(f"{'Stage':<8} {'numeric_em':>12} {'span_match':>12} {'combined':>10}")
    print("-" * 46)
    for stage, data in results.items():
        m = data["metrics"]
        print(f"{stage:<8} {m['numeric_em']:>12.4f} {m['span_match']:>12.4f} {m['combined']:>10.4f}")
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
