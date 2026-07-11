from __future__ import annotations

import torch

from optiz_qwen.scheduling import run_greedy_prefill_decode


class FakeOutputs:
    def __init__(self, logits, past_key_values=None) -> None:
        self.logits = logits
        self.past_key_values = past_key_values


class FakeGreedyModel:
    def __init__(self) -> None:
        self.calls = []
        self.vocab_size = 8

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"]
        step = input_ids.shape[-1]
        logits = torch.full((1, step, self.vocab_size), -1000.0)
        logits[0, -1, step % self.vocab_size] = 1000.0
        return FakeOutputs(logits=logits, past_key_values={"step": step})


def test_run_greedy_prefill_decode_uses_past_key_values() -> None:
    model = FakeGreedyModel()
    inputs = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
    }

    generated_ids, stats = run_greedy_prefill_decode(
        model,
        inputs,
        max_new_tokens=3,
        tokenizer=object(),
    )

    assert generated_ids.shape == (1, 3)
    assert stats.prompt_tokens == 3
    assert stats.generated_tokens == 3
    assert len(model.calls) == 3
    assert model.calls[0]["use_cache"] is True
    assert model.calls[1]["past_key_values"] == {"step": 3}
    assert model.calls[2]["past_key_values"] == {"step": 1}
