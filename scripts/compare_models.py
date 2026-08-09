"""Compare base, SFT and DPO models on the held-out validation set.

Generates predictions from each model on identical prompts, scores them with
the offline harness, logs metrics to MLflow as nested runs, and prints a
comparison table. Models are loaded sequentially to fit limited VRAM.

Usage:
    python scripts/compare_models.py \
        --config configs/post_training.yaml \
        --sft-adapter outputs/qwen3-financial-sft \
        --dpo-adapter outputs/qwen3-financial-dpo \
        --limit 50
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
from finpost.evaluate import score


def jsonl_records(path: str) -> list[dict]:
    """Read newline-delimited JSON records."""
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


@torch.inference_mode()
def generate_predictions(
    model_path: str,
    records: list[dict],
    max_seq_length: int,
    max_new_tokens: int = 64,
) -> list[str]:
    """Load a model, generate answers for each validation prompt, then release VRAM."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    predictions = []
    for record in records:
        messages = record["messages"][:-1]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        prediction = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        predictions.append(prediction.strip())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--dpo-adapter", required=True)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--output", default="outputs/comparison_results.json")
    args = parser.parse_args()

    config = load_config(args.config)
    records = jsonl_records(str(config.data.validation_path))[: args.limit]
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
            )
            metrics = score(predictions, golds)
            results[stage] = {"predictions": predictions, "metrics": metrics}
            with mlflow.start_run(run_name=stage, nested=True):
                mlflow.log_param("model_path", model_path)
                mlflow.log_param("n_eval", len(records))
                mlflow.log_metrics(metrics)
                pred_path = Path(f"outputs/predictions_{stage}.jsonl")
                pred_path.parent.mkdir(parents=True, exist_ok=True)
                pred_path.write_text(
                    "\n".join(json.dumps(p) for p in predictions),
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
