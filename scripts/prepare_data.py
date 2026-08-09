"""Prepare SFT and preference datasets from the raw TAT-QA JSON."""

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from finpost.config import load_config
from finpost.data import preference_record, sft_record, split_records, write_jsonl


def load_tatqa_json(dataset_name: str, split: str) -> list[dict]:
    """Download and parse TAT-QA without Arrow schema inference."""
    filename = f"tatqa_dataset_{split}.json"
    local_path = hf_hub_download(repo_id=dataset_name, filename=filename, repo_type="dataset")
    with open(local_path, encoding="utf-8") as stream:
        payload = json.load(stream)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in (split, "data", "examples"):
            if isinstance(payload.get(key), list):
                return payload[key]
    raise ValueError(f"Unexpected TAT-QA JSON structure in {filename}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    raw = load_tatqa_json(config.data.dataset_name, "train")
    records = []
    for example in raw:
        for question in example.get("questions", []):
            record = sft_record(example, question, config.data.max_context_chars)
            if record:
                records.append(record)
            if config.data.max_samples and len(records) >= config.data.max_samples:
                break
        if config.data.max_samples and len(records) >= config.data.max_samples:
            break
    if not records:
        raise ValueError("No usable TAT-QA records were produced")
    train, validation = split_records(records, config.data.validation_fraction, config.seed)
    preferences = [preference_record(record, "I cannot determine this from the context.") for record in train]
    write_jsonl(train, config.data.train_path)
    write_jsonl(validation, config.data.validation_path)
    write_jsonl(preferences, config.data.preference_path)
    print(f"Prepared {len(train)} training, {len(validation)} validation and {len(preferences)} preference records")


if __name__ == "__main__":
    main()
