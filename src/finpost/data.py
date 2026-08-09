"""Pure data transformations for chat and preference records."""

from __future__ import annotations

import json
import random
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


def preference_record(sft: dict, rejected: str) -> dict:
    """Convert an SFT record into TRL preference format."""
    return {
        "prompt": sft["messages"][:-1],
        "chosen": sft["messages"][-1]["content"],
        "rejected": rejected,
    }


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
