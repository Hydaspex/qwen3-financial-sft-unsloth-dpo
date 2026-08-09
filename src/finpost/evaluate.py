"""Small, reproducible evaluation metrics for financial answers."""

import re

_NUMBER = re.compile(r"-?\d[\d,]*\.?\d*")


def first_number(text: str) -> float | None:
    """Extract the first numeric token from model output."""
    match = _NUMBER.search(text)
    return float(match.group().replace(",", "")) if match else None


def numeric_match(prediction: str, gold: str, relative_tolerance: float = 1e-3) -> bool:
    """Compare the first numeric tokens using relative tolerance."""
    predicted, expected = first_number(prediction), first_number(gold)
    if predicted is None or expected is None:
        return False
    if expected == 0:
        return abs(predicted) < relative_tolerance
    return abs(predicted - expected) / abs(expected) <= relative_tolerance


def span_match(prediction: str, gold: str) -> bool:
    """Perform case-insensitive containment matching for text answers."""
    return gold.strip().casefold() in prediction.strip().casefold()


def score(predictions: list[str], golds: list[str]) -> dict[str, float]:
    """Return numeric, span and combined metrics."""
    if len(predictions) != len(golds) or not golds:
        raise ValueError("predictions and golds must have the same non-zero length")
    numeric = sum(numeric_match(p, g) for p, g in zip(predictions, golds)) / len(golds)
    span = sum(span_match(p, g) for p, g in zip(predictions, golds)) / len(golds)
    return {"numeric_em": numeric, "span_match": span, "combined": (numeric + span) / 2}
