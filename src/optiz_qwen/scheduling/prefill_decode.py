"""Prefill/decode runtime helpers for opt-in generation optimization."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class PrefillDecodeStats:
    """Timing and execution metadata for a greedy prefill/decode run."""

    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float
    ttft_seconds: float
    elapsed_seconds: float


def run_greedy_prefill_decode(
    model: Any,
    inputs: dict[str, Any],
    *,
    max_new_tokens: int,
    tokenizer: Any,
    eos_token_id: int | None = None,
    kv_cache: Any | None = None,
    post_prefill_callback: Any | None = None,
    post_decode_callback: Any | None = None,
) -> tuple[torch.Tensor, PrefillDecodeStats]:
    """Run a greedy prefill + token-by-token decode loop.

    The helper is intentionally narrow: batch size 1, greedy decoding only,
    and opt-in use from the Qwen3.5-2B wrapper where baseline behavior must
    remain unchanged.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    input_ids = inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("inputs must include input_ids.")
    if int(input_ids.shape[0]) != 1:
        raise ValueError("run_greedy_prefill_decode only supports batch size 1.")

    device = input_ids.device
    prompt_tokens = int(input_ids.shape[-1])
    prefill_inputs = dict(inputs)
    prefill_inputs["use_cache"] = True
    defer_prefill_cache = bool(
        kv_cache is not None and getattr(kv_cache, "defer_prefill_cache_injection", False)
    )
    activation_threshold = int(
        getattr(getattr(kv_cache, "qserve_config", None), "activation_threshold", 0)
    )
    decode_warmup_tokens = int(
        getattr(getattr(kv_cache, "qserve_config", None), "decode_warmup_tokens", 0)
    )
    # Multiple-choice VLM answers often decide their option in the first few
    # decode tokens.  Preserve those decision-critical tokens exactly on the
    # native cache, then use packed KV only for a sufficiently long tail.
    activate_after_prefill = (
        defer_prefill_cache
        and prompt_tokens >= activation_threshold
        and decode_warmup_tokens <= 1
    )
    pending_deferred_activation = defer_prefill_cache and not activate_after_prefill
    if kv_cache is not None:
        if not defer_prefill_cache:
            prefill_inputs["past_key_values"] = kv_cache
        attention_mask = inputs.get("attention_mask")
        dense_decode_mask = attention_mask is None or bool(torch.all(attention_mask == 1).item())
        setattr(kv_cache, "_optiz_dense_decode_mask", dense_decode_mask)

    _synchronize_device(input_ids)
    prefill_start = time.perf_counter()
    with torch.no_grad():
        prefill_outputs = model(**prefill_inputs)
    _synchronize_device(input_ids)
    prefill_end = time.perf_counter()

    past_key_values = getattr(prefill_outputs, "past_key_values", None)
    logits = prefill_outputs.logits[:, -1, :]
    next_token = torch.argmax(logits, dim=-1)
    generated_ids = [int(next_token.item())]
    first_token_at = time.perf_counter()

    if eos_token_id is not None and generated_ids[-1] == eos_token_id:
        elapsed = first_token_at - prefill_start
        return _to_token_tensor(generated_ids, device), PrefillDecodeStats(
            prompt_tokens=prompt_tokens,
            generated_tokens=len(generated_ids),
            prefill_seconds=prefill_end - prefill_start,
            decode_seconds=first_token_at - prefill_end,
            ttft_seconds=elapsed,
            elapsed_seconds=elapsed,
        )

    if activate_after_prefill:
        if past_key_values is None:
            raise RuntimeError("native prefill did not return a cache for deferred qserve decode.")
        # This bookkeeping is required for decode, but it is not part of the
        # model's first-token computation and must not contaminate TTFT.
        kv_cache.adopt_native_prefill_cache(past_key_values)
        past_key_values = kv_cache
    if post_prefill_callback is not None:
        post_prefill_callback()

    decode_start = first_token_at
    attention_mask = inputs.get("attention_mask")
    current_input_ids = next_token.to(device=device, dtype=torch.long).view(1, 1)
    if attention_mask is not None:
        # The first decode step consumes the token predicted from the prefill
        # pass, so its mask must already extend by one position.
        attention_mask = torch.cat(
            [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
            dim=-1,
        )

    for _ in range(max_new_tokens - 1):
        decode_kwargs = _build_decode_kwargs(
            model=model,
            input_ids=current_input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        with torch.no_grad():
            decode_outputs = model(**decode_kwargs)
        past_key_values = getattr(decode_outputs, "past_key_values", past_key_values)
        if (
            pending_deferred_activation
            and past_key_values is not None
            and prompt_tokens + len(generated_ids) >= activation_threshold
            and len(generated_ids) >= decode_warmup_tokens
        ):
            # Until this point the candidate is byte-for-byte on the native
            # cache path.  Only adopt it once the real request reaches the
            # experimentally selected packed-KV crossover.
            kv_cache.adopt_native_prefill_cache(past_key_values)
            past_key_values = kv_cache
            pending_deferred_activation = False
        if post_decode_callback is not None:
            post_decode_callback()
        logits = decode_outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1)
        token_id = int(next_token.item())
        generated_ids.append(token_id)
        if eos_token_id is not None and token_id == eos_token_id:
            break
        current_input_ids = next_token.to(device=device, dtype=torch.long).view(1, 1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )

    end = time.perf_counter()
    return _to_token_tensor(generated_ids, device), PrefillDecodeStats(
        prompt_tokens=prompt_tokens,
        generated_tokens=len(generated_ids),
        prefill_seconds=prefill_end - prefill_start,
        decode_seconds=end - decode_start,
        ttft_seconds=first_token_at - prefill_start,
        elapsed_seconds=end - prefill_start,
    )


def _synchronize_device(tensor: torch.Tensor) -> None:
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _build_decode_kwargs(
    *,
    model: Any,
    input_ids: torch.Tensor,
    past_key_values: Any,
    attention_mask: torch.Tensor | None,
) -> dict[str, Any]:
    prepare = getattr(model, "prepare_inputs_for_generation", None)
    position_id_prepare = getattr(model, "_prepare_position_ids_for_generation", None)
    position_ids = None
    if callable(position_id_prepare):
        try:
            position_ids = position_id_prepare(
                input_ids,
                {
                    "input_ids": input_ids,
                    "past_key_values": past_key_values,
                    "attention_mask": attention_mask,
                },
            )
        except Exception:
            position_ids = None
    if callable(prepare):
        try:
            prepared = prepare(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
            )
            if isinstance(prepared, dict):
                return prepared
        except Exception:
            pass
    decode_kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": True,
    }
    if attention_mask is not None:
        decode_kwargs["attention_mask"] = attention_mask
    if position_ids is not None:
        decode_kwargs["position_ids"] = position_ids
    return decode_kwargs


def _to_token_tensor(token_ids: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor([token_ids], device=device, dtype=torch.long)
