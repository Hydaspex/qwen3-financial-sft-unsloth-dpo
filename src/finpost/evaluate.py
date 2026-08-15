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


NUMERIC_ANSWER_TYPES = frozenset({"arithmetic", "count"})


def match_by_type(prediction: str, gold: str, answer_type: str) -> bool:
    """Score an answer with the metric its type actually calls for.

    Applying both metrics to every example is not a stricter test, it is a
    broken one: a numeric metric on a textual answer can only ever fail, so
    each metric's ceiling collapses to the share of the split it suits.
    Multi-span answers are checked span by span, since requiring the joined
    string verbatim tests comma placement rather than comprehension.
    """
    if answer_type in NUMERIC_ANSWER_TYPES:
        return numeric_match(prediction, gold)
    if answer_type == "multi-span":
        parts = [part.strip() for part in gold.split(",") if part.strip()]
        return bool(parts) and all(span_match(prediction, part) for part in parts)
    return span_match(prediction, gold)


def score_by_type(
    predictions: list[str], golds: list[str], answer_types: list[str]
) -> dict[str, float]:
    """Return overall accuracy plus a per-answer-type breakdown.

    The breakdown is the point: a single headline number hides that this
    split is ~43% span and ~42% arithmetic, which are different skills.
    """
    results = per_example_results(predictions, golds, answer_types)
    metrics = {"accuracy": sum(r["correct"] for r in results) / len(results)}
    for answer_type in sorted({r["answer_type"] for r in results}):
        subset = [r["correct"] for r in results if r["answer_type"] == answer_type]
        key = answer_type.replace("-", "_") or "unknown"
        metrics[f"accuracy_{key}"] = sum(subset) / len(subset)
        metrics[f"n_{key}"] = float(len(subset))
    return metrics


def per_example_results(
    predictions: list[str], golds: list[str], answer_types: list[str] | None = None
) -> list[dict]:
    """Return per-example prediction/gold/correctness records.

    Used to persist enough detail per example to support bootstrap CIs,
    paired significance tests and error analysis downstream. Supplying
    answer_types adds the type-aware "correct" verdict used for headline
    accuracy; the raw metric columns stay for backwards comparison.
    """
    if len(predictions) != len(golds) or not golds:
        raise ValueError("predictions and golds must have the same non-zero length")
    if answer_types is not None and len(answer_types) != len(golds):
        raise ValueError("answer_types must align with predictions and golds")
    types = answer_types if answer_types is not None else [""] * len(golds)
    return [
        {
            "prediction": prediction,
            "gold": gold,
            "answer_type": answer_type,
            "numeric_em": numeric_match(prediction, gold),
            "span_match": span_match(prediction, gold),
            "correct": match_by_type(prediction, gold, answer_type),
        }
        for prediction, gold, answer_type in zip(predictions, golds, types)
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
