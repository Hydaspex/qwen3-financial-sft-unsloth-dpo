"""Prepare SFT and preference datasets from TAT-QA."""

import argparse

from datasets import load_dataset

from finpost.config import load_config
from finpost.data import preference_record, sft_record, split_records, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    raw = load_dataset(config.data.dataset_name, split="train")
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
    train, validation = split_records(records, config.data.validation_fraction, config.seed)
    preferences = [preference_record(record, "I cannot determine this from the context.") for record in train]
    write_jsonl(train, config.data.train_path)
    write_jsonl(validation, config.data.validation_path)
    write_jsonl(preferences, config.data.preference_path)
    print(f"Prepared {len(train)} training, {len(validation)} validation and {len(preferences)} preference records")


if __name__ == "__main__":
    main()
