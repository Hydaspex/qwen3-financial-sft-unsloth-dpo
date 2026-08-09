"""Validated YAML configuration for the SFT-to-DPO pipeline."""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name_or_path: str
    max_seq_length: int = 2048
    load_in_4bit: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class DataConfig(BaseModel):
    dataset_name: str
    train_path: Path
    validation_path: Path
    preference_path: Path
    max_context_chars: int = 6000
    max_samples: int | None = None
    validation_fraction: float = Field(default=0.05, gt=0, lt=0.5)


class StageConfig(BaseModel):
    output_dir: Path
    epochs: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float
    warmup_ratio: float = 0.03
    max_steps: int = -1
    beta: float = 0.1
    max_length: int = 2048
    max_prompt_length: int = 1536


class MLflowConfig(BaseModel):
    tracking_uri: str | None = None
    experiment: str


class PipelineConfig(BaseModel):
    experiment_name: str
    seed: int = 42
    model: ModelConfig
    data: DataConfig
    sft: StageConfig
    dpo: StageConfig
    mlflow: MLflowConfig


def load_config(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline configuration."""
    with open(path, encoding="utf-8") as stream:
        return PipelineConfig.model_validate(yaml.safe_load(stream))
