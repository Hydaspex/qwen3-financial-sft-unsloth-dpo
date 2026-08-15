import pytest

from finpost.checkpoints import find_resumable_checkpoint
from finpost.data import (
    preference_record,
    preference_records,
    prompt_completion_record,
    split_records,
)
from finpost.evaluate import (
    bootstrap_ci,
    match_by_type,
    mcnemar_one_sided,
    numeric_match,
    per_example_results,
    score,
    score_by_type,
)
from finpost.generate import batch_generate


def test_preference_record_shape():
    sft = {"messages": [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "42"}]}
    record = preference_record(sft, "I do not know")
    assert record["chosen"] == "42"
    assert record["rejected"] == "I do not know"
    assert record["prompt"][0]["role"] == "user"


def test_prompt_completion_record_separates_prompt_from_answer():
    """TRL enables completion-only loss only when it sees separate "prompt"
    and "completion" columns; a single pre-rendered text field makes it train
    on every token, so the report context (~99% of them) dominates the
    gradient instead of the answer."""
    sft = {"messages": [
        {"role": "system", "content": "You are a financial analyst."},
        {"role": "user", "content": "Table...\n\nQuestion: What is revenue?"},
        {"role": "assistant", "content": "435"},
    ]}

    record = prompt_completion_record(sft)

    assert set(record) == {"prompt", "completion"}
    assert record["completion"] == [{"role": "assistant", "content": "435"}]
    assert [m["role"] for m in record["prompt"]] == ["system", "user"]


def _chat(answer: str) -> dict:
    return {"messages": [
        {"role": "user", "content": "Question?"},
        {"role": "assistant", "content": answer},
    ]}


def test_preference_records_reject_a_different_answer():
    """A constant rejection only teaches the model to avoid one phrase; the
    rejected answer has to be a real alternative for the pair to carry signal."""
    records = [_chat("435"), _chat("1.8"), _chat("30,927"), _chat("2018")]

    preferences = preference_records(records, seed=42)

    assert len(preferences) == len(records)
    for pref in preferences:
        assert pref["rejected"] != pref["chosen"]


def test_preference_records_match_answer_shape():
    """Figures are rejected with figures and prose with prose, so the pair
    turns on correctness rather than on which one looks like an answer."""
    records = [_chat("435"), _chat("1.8"), _chat("restricted cash"), _chat("goodwill impairment")]

    preferences = preference_records(records, seed=0)

    numeric_rejections = [p["rejected"] for p in preferences if p["chosen"] in {"435", "1.8"}]
    assert all(r in {"435", "1.8"} for r in numeric_rejections)


def test_preference_records_are_deterministic():
    records = [_chat("435"), _chat("1.8"), _chat("30,927")]
    assert preference_records(records, seed=7) == preference_records(records, seed=7)


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
    assert [r["numeric_em"] for r in results] == [True, False]
    assert [r["span_match"] for r in results] == [True, True]
    assert [r["prediction"] for r in results] == ["1000", "cloud services"]


def test_match_by_type_routes_to_the_right_metric():
    """The same prediction is right or wrong depending on the question type,
    which is exactly what scoring every example with both metrics destroys."""
    # An arithmetic answer is judged numerically, so prose around it is fine.
    assert match_by_type("The total is 1,000", "1000", "arithmetic")
    # A span answer is judged textually, so a number alone cannot satisfy it.
    assert not match_by_type("1000", "cloud services", "span")
    assert match_by_type("we sell cloud services", "Cloud Services", "span")


def test_match_by_type_multi_span_requires_every_span():
    assert match_by_type("2019 and 2018", "2019, 2018", "multi-span")
    assert not match_by_type("only 2019 here", "2019, 2018", "multi-span")


def test_score_by_type_breaks_down_by_answer_type():
    predictions = ["1000", "cloud services", "50", "goodwill"]
    golds = ["1000", "Cloud Services", "999", "goodwill"]
    types = ["arithmetic", "span", "arithmetic", "span"]

    metrics = score_by_type(predictions, golds, types)

    assert metrics["accuracy"] == 0.75
    assert metrics["accuracy_arithmetic"] == 0.5
    assert metrics["accuracy_span"] == 1.0
    assert metrics["n_arithmetic"] == 2.0


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


def _tiny_causal_lm():
    """A ~few-hundred-KB public test model, standard practice for testing
    generation code without needing a real model (the same models
    transformers' own test suite uses for this purpose)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "hf-internal-testing/tiny-random-gpt2"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    return model, tokenizer


def test_batch_generate_matches_single_example_generation():
    """Greedy decoding must give identical output whether a prompt is
    generated alone or padded into a batch with others — this is exactly the
    left-padding/slicing correctness compare_models.py's docstring flags as a
    risk, verified here on CPU instead of only by hand on a GPU."""
    model, tokenizer = _tiny_causal_lm()
    prompts = ["Hello there", "A much longer prompt to force real padding", "Hi"]

    solo = [
        batch_generate(model, tokenizer, [p], max_seq_length=32, max_new_tokens=5, batch_size=1)[0]
        for p in prompts
    ]
    batched = batch_generate(
        model, tokenizer, prompts, max_seq_length=32, max_new_tokens=5, batch_size=len(prompts)
    )

    assert batched == solo


def test_batch_generate_sets_left_truncation():
    """truncation_side must be left, not HF's default right: right truncation
    would cut off the chat template's trailing generation-prompt marker on
    any prompt longer than max_seq_length, leaving the model no signal to
    start responding — observed in practice as empty/degenerate output on
    long-context examples."""
    model, tokenizer = _tiny_causal_lm()

    batch_generate(model, tokenizer, ["hello"], max_seq_length=32, max_new_tokens=3, batch_size=1)

    assert tokenizer.truncation_side == "left"


def test_batch_generate_handles_batch_size_smaller_than_input():
    model, tokenizer = _tiny_causal_lm()
    prompts = ["one", "two", "three", "four", "five"]

    predictions = batch_generate(
        model, tokenizer, prompts, max_seq_length=32, max_new_tokens=3, batch_size=2
    )

    assert len(predictions) == len(prompts)


def test_find_resumable_checkpoint_missing_dir_returns_none(tmp_path):
    assert find_resumable_checkpoint(tmp_path / "does-not-exist") is None


def test_find_resumable_checkpoint_empty_dir_returns_none(tmp_path):
    assert find_resumable_checkpoint(tmp_path) is None


def test_find_resumable_checkpoint_returns_highest_step(tmp_path):
    (tmp_path / "checkpoint-50").mkdir()
    (tmp_path / "checkpoint-200").mkdir()
    (tmp_path / "checkpoint-100").mkdir()

    result = find_resumable_checkpoint(tmp_path)

    assert result == str(tmp_path / "checkpoint-200")


def test_find_resumable_checkpoint_ignores_non_checkpoint_dirs(tmp_path):
    (tmp_path / "checkpoint-10").mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "some_file.txt").write_text("not a checkpoint")

    result = find_resumable_checkpoint(tmp_path)

    assert result == str(tmp_path / "checkpoint-10")
