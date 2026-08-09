"""DPO preference optimisation entry point."""

import argparse

import mlflow
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import DPOConfig, DPOTrainer

from finpost.config import load_config


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
    dataset = load_dataset("json", data_files=str(config.data.preference_path), split="train")
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
