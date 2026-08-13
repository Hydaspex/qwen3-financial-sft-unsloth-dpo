"""Checkpoint discovery for resumable training.

Kept as a pure, CPU-testable function even though it wraps a transformers
utility: get_last_checkpoint requires its folder argument to already exist,
which it won't on a fresh run, so the missing-directory guard is real logic
worth verifying without needing a GPU or an actual trained checkpoint's
contents — only directories named checkpoint-N.
"""

from pathlib import Path

from transformers.trainer_utils import get_last_checkpoint


def find_resumable_checkpoint(output_dir: str | Path) -> str | None:
    """Return the path to the most recent checkpoint in output_dir, or None.

    Safe to call before output_dir exists (a fresh run) or when it exists but
    holds no checkpoints yet.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    return get_last_checkpoint(str(output_dir))
