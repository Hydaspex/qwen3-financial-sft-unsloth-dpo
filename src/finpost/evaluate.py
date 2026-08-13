"""Small, reproducible evaluation metrics for financial answers."""

import random
import re
from collections.abc import Sequence
from math import erf, sqrt

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


def per_example_results(predictions: list[str], golds: list[str]) -> list[dict]:
    """Return per-example prediction/gold/correctness records.

    Used to persist enough detail per example to support bootstrap CIs,
    paired significance tests and error analysis downstream.
    """
    if len(predictions) != len(golds) or not golds:
        raise ValueError("predictions and golds must have the same non-zero length")
    return [
        {
            "prediction": prediction,
            "gold": gold,
            "numeric_em": numeric_match(prediction, gold),
            "span_match": span_match(prediction, gold),
        }
        for prediction, gold in zip(predictions, golds)
    ]


def bootstrap_ci(
    values: Sequence[bool],
    n_boot: int = 1000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean of a per-example metric vector.

    Returns (point_estimate, ci_low, ci_high).
    """
    values = list(values)
    if not values:
        raise ValueError("values must be non-empty")
    n = len(values)
    point = sum(values) / n
    rng = random.Random(seed)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    alpha = (1 - confidence) / 2
    lo_idx = max(int(alpha * n_boot), 0)
    hi_idx = min(int((1 - alpha) * n_boot) - 1, n_boot - 1)
    return point, means[lo_idx], means[hi_idx]


def mcnemar_one_sided(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, float]:
    """One-sided McNemar test: is B better than A on paired per-example outcomes?

    correct_a/correct_b must be aligned (same examples, same order) boolean
    vectors, e.g. per-example numeric_em or span_match columns for two
    configs evaluated on an identical validation slice.
    """
    if len(correct_a) != len(correct_b) or not correct_a:
        raise ValueError("correct_a and correct_b must have the same non-zero length")
    n10 = sum(a and not b for a, b in zip(correct_a, correct_b))
    n01 = sum(b and not a for a, b in zip(correct_a, correct_b))
    n_discordant = n10 + n01
    if n_discordant == 0:
        return {"n10": n10, "n01": n01, "statistic": 0.0, "p_value": 1.0}
    statistic = (abs(n01 - n10) - 1) ** 2 / n_discordant
    z = (n01 - n10) / sqrt(n_discordant)
    p_one_sided = (1 - erf(abs(z) / sqrt(2))) / 2 if n01 > n10 else 1.0
    return {"n10": n10, "n01": n01, "statistic": statistic, "p_value": p_one_sided}
