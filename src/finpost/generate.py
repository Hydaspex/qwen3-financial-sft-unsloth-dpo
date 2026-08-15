"""Batched greedy generation for causal language models.

Deliberately imports only torch, not unsloth: it takes an already-loaded
model/tokenizer rather than a model path, so it stays usable (and unit
testable on CPU with any small HF model) independent of how the model was
loaded. Unsloth-specific loading lives in the scripts that call this.
"""

import torch


@torch.inference_mode()
def batch_generate(
    model,
    tokenizer,
    prompts: list[str],
    max_seq_length: int,
    max_new_tokens: int = 64,
    batch_size: int = 8,
) -> list[str]:
    """Greedily generate a completion per prompt, batched.

    This is the part of generation with real correctness risk: padding side,
    slicing out only the newly generated tokens, and pad-token handling. A
    bug here silently corrupts predictions rather than raising, which is why
    it is kept separate from model loading and covered by
    tests/test_pipeline.py::test_batch_generate_matches_single_example_generation
    using a tiny public HF model, rather than only spot-checked by hand on a
    GPU.
    """
    # Left padding is required for batched causal-LM generation: with every
    # row's real tokens right-aligned, generate() appends new tokens after
    # the same index (the padded prompt length) for every row, so a single
    # prompt_len slice below is correct for the whole batch.
    tokenizer.padding_side = "left"
    # Left truncation too: the chat template's generation-prompt marker sits
    # at the very end of the rendered prompt. Right truncation (HF's default)
    # would cut that marker off on any prompt longer than max_seq_length,
    # leaving the model with no clear signal to start responding -- observed
    # in practice as empty or degenerate repeating output on long-context
    # examples.
    tokenizer.truncation_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    predictions: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_length,
        ).to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        prompt_len = inputs["input_ids"].shape[1]
        for row in outputs:
            prediction = tokenizer.decode(row[prompt_len:], skip_special_tokens=True)
            predictions.append(prediction.strip())

    return predictions
