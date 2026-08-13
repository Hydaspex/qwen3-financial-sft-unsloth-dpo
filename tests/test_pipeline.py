import pytest

from finpost.data import preference_record, split_records
from finpost.evaluate import (
    bootstrap_ci,
    mcnemar_one_sided,
    numeric_match,
    per_example_results,
    score,
)


def test_preference_record_shape():
    sft = {"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "42"}]}
    record = preference_record(sft, "I do not know")
    assert record["chosen"] == "42"
    assert record["rejected"] == "I do not know"
    assert record["prompt"][0]["role"] == "user"


def test_split_is_deterministic():
    records = [{"id": i} for i in range(20)]
    first = split_records(records, 0.2, 42)
    second = split_records(records, 0.2, 42)
    assert first == second
    assert {r["id"] for r in first[0]}.isdisjoint({r["id"] for r in first[1]})


def test_numeric_and_score_metrics():
    assert numeric_match("approximately 1,000", "1000")
    metrics = score(["1000", "cloud services"], ["1000", "Cloud Services"])
    assert metrics == {"numeric_em": 0.5, "span_match": 1.0, "combined": 0.75}


def test_empty_score_rejected():
    with pytest.raises(ValueError):
        score([], [])


def test_per_example_results_shape():
    results = per_example_results(["1000", "cloud services"], ["1000", "Cloud Services"])
    assert results == [
        {"prediction": "1000", "gold": "1000", "numeric_em": True, "span_match": True},
        {"prediction": "cloud services", "gold": "Cloud Services", "numeric_em": False, "span_match": True},
    ]


def test_per_example_results_rejects_length_mismatch():
    with pytest.raises(ValueError):
        per_example_results(["a"], [])


def test_bootstrap_ci_all_correct_is_tight():
    point, lo, hi = bootstrap_ci([True] * 20, n_boot=200, seed=1)
    assert point == 1.0
    assert lo == 1.0
    assert hi == 1.0


def test_bootstrap_ci_is_deterministic_given_seed():
    values = [True, False, True, True, False, False, True, False]
    first = bootstrap_ci(values, n_boot=200, seed=7)
    second = bootstrap_ci(values, n_boot=200, seed=7)
    assert first == second


def test_bootstrap_ci_bounds_contain_point_estimate():
    values = [True, False, True, True, False]
    point, lo, hi = bootstrap_ci(values, n_boot=500, seed=3)
    assert lo <= point <= hi


def test_bootstrap_ci_rejects_empty():
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_mcnemar_detects_directional_improvement():
    correct_a = [True, True, False, False, False, False, True, False]
    correct_b = [True, True, True, True, True, False, True, False]
    result = mcnemar_one_sided(correct_a, correct_b)
    assert result["n01"] > result["n10"]
    assert result["p_value"] < 0.5


def test_mcnemar_no_discordant_pairs_returns_p_one():
    result = mcnemar_one_sided([True, False], [True, False])
    assert result["p_value"] == 1.0


def test_mcnemar_rejects_length_mismatch():
    with pytest.raises(ValueError):
        mcnemar_one_sided([True], [True, False])
