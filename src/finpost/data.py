"""Pure data transformations for chat and preference records."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = (
    "You are a financial analyst. Use only the supplied report context. "
    "Give a concise answer and do not invent evidence."
)


def context_text(example: dict[str, Any], max_chars: int) -> str:
    """Combine report paragraphs and table rows into bounded context."""
    parts = [p.get("text", "") for p in example.get("paragraphs", [])]
    table = example.get("table", {})
    rows = table.get("table", []) if isinstance(table, dict) else table
    parts.extend(" | ".join(map(str, row)) for row in rows)
    return "\n\n".join(p for p in parts if p)[:max_chars]


def answer_text(answer: Any) -> str:
    """Convert scalar and multi-span answers to a training string."""
    return ", ".join(map(str, answer)) if isinstance(answer, list) else str(answer)


def sft_record(example: dict[str, Any], question: dict[str, Any], max_chars: int) -> dict | None:
    """Build one assistant-completion record from a TAT-QA question."""
    context = context_text(example, max_chars)
    if not context or question.get("answer") in (None, ""):
        return None
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}\n\nQuestion: {question['question']}"},
        {"role": "assistant", "content": answer_text(question["answer"])},
    ]}


def prompt_completion_record(sft: dict) -> dict:
    """Split a chat record into TRL's conversational prompt-completion format.

    TRL decides whether to mask the prompt by inspecting the dataset columns:
    it enables completion-only loss only when both "prompt" and "completion"
    are present, and otherwise treats the row as plain language modelling and
    trains on every token. Handing it a single pre-rendered text field
    therefore silently trains the model to reproduce the report context --
    which here is ~99% of the tokens -- instead of to answer the question.
    """
    return {"prompt": sft["messages"][:-1], "completion": sft["messages"][-1:]}


def preference_record(sft: dict, rejected: str) -> dict:
    """Convert an SFT record into TRL preference format."""
    return {
        "prompt": sft["messages"][:-1],
        "chosen": sft["messages"][-1]["content"],
        "rejected": rejected,
    }


def _looks_numeric(text: str) -> bool:
    """True if the answer is a bare figure rather than prose."""
    return bool(re.fullmatch(r"[-+$(]?\s*[\d,]+\.?\d*\s*[)%]?", text.strip()))


def preference_records(records: list[dict], seed: int) -> list[dict]:
    """Pair each record with a wrong answer of the same shape.

    A single constant rejection ("I cannot determine...") only teaches the
    model not to emit that one phrase, which is why DPO barely moved the
    metrics. Drawing the rejection from another question's gold answer --
    and matching figure against figure, prose against prose -- makes the
    pair turn on whether the answer is *right*, not on which one looks like
    an answer at all.
    """
    golds = [record["messages"][-1]["content"] for record in records]
    numeric = [g for g in golds if _looks_numeric(g)]
    prose = [g for g in golds if not _looks_numeric(g)]
    rng = random.Random(seed)

    preferences = []
    for record, gold in zip(records, golds):
        pool = numeric if _looks_numeric(gold) else prose
        # Fall back to the other pool if this one is too small to differ.
        candidates = pool if len(pool) > 1 else golds
        rejected = gold
        for _ in range(10):
            rejected = rng.choice(candidates)
            if rejected.strip().casefold() != gold.strip().casefold():
                break
        else:
            continue  # no distinct distractor found; drop rather than emit a degenerate pair
        preferences.append(preference_record(record, rejected))
    return preferences


def split_records(records: list[dict], fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Return a deterministic train/validation split."""
    indices = list(range(len(records)))
    random.Random(seed).shuffle(indices)
    n_validation = max(1, int(len(records) * fraction))
    validation = set(indices[:n_validation])
    return (
        [r for i, r in enumerate(records) if i not in validation],
        [r for i, r in enumerate(records) if i in validation],
    )


def write_jsonl(records: list[dict], path: str | Path) -> None:
    """Write records as newline-delimited JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as stream:
        stream.writelines(json.dumps(record) + "\n" for record in records)
