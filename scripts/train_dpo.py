"""DPO preference optimisation entry point."""

import argparse
import json
import sys
from pathlib import Path

import mlflow
import torch
import unsloth  # noqa: F401
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finpost.config import load_config


def precision_flags() -> tuple[bool, bool]:
    """Return bf16 and fp16 flags appropriate for the active GPU."""
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16
    return bf16, fp16


def jsonl_records(path: str) -> list[dict]:
    """Read newline-delimited JSON records."""
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def render_preference_pairs(records: list[dict], tokenizer) -> list[dict]:
    """Render preference records into plain prompt/chosen/rejected strings.

    The prompt uses the model's chat template with a generation prompt so the
    completions are scored as continuations. Rendering before trainer
    construction avoids heterogeneous schemas between map shards.
    """
    rendered = []
    for record in records:
        prompt = record["prompt"]
        if isinstance(prompt, list):
            prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
            )
        rendered.append({
            "prompt": prompt,
            "chosen": str(record["chosen"]),
            "rejected": str(record["rejected"]),
        })
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sft-adapter", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.sft_adapter,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
    )
    pairs = render_preference_pairs(jsonl_records(str(config.data.preference_path)), tokenizer)
    dataset = Dataset.from_list(pairs)
    use_bf16, use_fp16 = precision_flags()
    print(f"Training precision: bf16={use_bf16}, fp16={use_fp16}")
    trainer = DPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            output_dir=str(config.dpo.output_dir),
            num_train_epochs=config.dpo.epochs,
            per_device_train_batch_size=config.dpo.batch_size,
            gradient_accumulation_steps=config.dpo.gradient_accumulation_steps,
            learning_rate=config.dpo.learning_rate,
            beta=config.dpo.beta,
            max_length=config.dpo.max_length,
            max_prompt_length=config.dpo.max_prompt_length,
            bf16=use_bf16,
            fp16=use_fp16,
            report_to=[],
        ),
    )
    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment)
    with mlflow.start_run(run_name=f"{config.experiment_name}-dpo"):
        trainer.train()
        trainer.save_model(str(config.dpo.output_dir))
        mlflow.log_param("stage", "dpo")
        mlflow.log_artifacts(str(config.dpo.output_dir), artifact_path="dpo_adapter")


if __name__ == "__main__":
    main()
