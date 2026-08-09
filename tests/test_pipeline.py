import pytest

from finpost.data import preference_record, split_records
from finpost.evaluate import numeric_match, score


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
