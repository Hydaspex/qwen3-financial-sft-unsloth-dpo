"""Unsloth QLoRA SFT entry point."""

import argparse
import json
import sys
from pathlib import Path

import mlflow
import torch
import unsloth  # noqa: F401
from datasets import Dataset
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer
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


def render_texts(records: list[dict], tokenizer) -> list[dict]:
    """Render chat records into a plain text field before batching."""
    rendered = []
    for record in records:
        if "text" in record:
            text = record["text"]
        else:
            text = tokenizer.apply_chat_template(
                record["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        rendered.append({"text": text})
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.model.name_or_path,
        max_seq_length=config.model.max_seq_length,
        load_in_4bit=config.model.load_in_4bit,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=config.model.lora_rank,
        lora_alpha=config.model.lora_alpha,
        lora_dropout=config.model.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    train = Dataset.from_list(render_texts(jsonl_records(str(config.data.train_path)), tokenizer))
    validation = Dataset.from_list(render_texts(jsonl_records(str(config.data.validation_path)), tokenizer))
    use_bf16, use_fp16 = precision_flags()
    print(f"Training precision: bf16={use_bf16}, fp16={use_fp16}")
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train,
        eval_dataset=validation,
        args=SFTConfig(
            output_dir=str(config.sft.output_dir),
            num_train_epochs=config.sft.epochs,
            per_device_train_batch_size=config.sft.batch_size,
            gradient_accumulation_steps=config.sft.gradient_accumulation_steps,
            learning_rate=config.sft.learning_rate,
            warmup_ratio=config.sft.warmup_ratio,
            max_length=config.model.max_seq_length,
            dataset_text_field="text",
            bf16=use_bf16,
            fp16=use_fp16,
            report_to=[],
            # A flaky free-tier Colab runtime can disconnect at any point in a
            # multi-hour run, so checkpoint often enough that a disconnect
            # costs minutes, not hours. save_total_limit keeps disk bounded.
            save_strategy="steps",
            save_steps=100,
            save_total_limit=3,
        ),
    )
    # Auto-resume: if output_dir already has a checkpoint (e.g. this script
    # was killed mid-run by a runtime disconnect), pick up from it instead of
    # silently restarting at step 0. get_last_checkpoint requires the folder
    # to already exist, which it won't on a fresh run.
    last_checkpoint = None
    if Path(config.sft.output_dir).exists():
        last_checkpoint = get_last_checkpoint(str(config.sft.output_dir))
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")

    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment)
    with mlflow.start_run(run_name=f"{config.experiment_name}-sft"):
        trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model(str(config.sft.output_dir))
        mlflow.log_param("stage", "sft")
        mlflow.log_artifacts(str(config.sft.output_dir), artifact_path="sft_adapter")


if __name__ == "__main__":
    main()
