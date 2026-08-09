"""Offline evaluation entry point for prediction and gold text files."""

import argparse

from finpost.config import load_config
from finpost.evaluate import score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--golds", required=True)
    args = parser.parse_args()
    load_config(args.config)
    with open(args.predictions, encoding="utf-8") as stream:
        predictions = [line.rstrip("\n") for line in stream]
    with open(args.golds, encoding="utf-8") as stream:
        golds = [line.rstrip("\n") for line in stream]
    print(score(predictions, golds))


if __name__ == "__main__":
    main()
