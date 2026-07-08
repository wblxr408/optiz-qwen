"""Participant wrapper contract for the DNDX public benchmark path."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class GenerationConfig:
    max_new_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class GenerationResult:
    text: str
    token_count: int
    ttft_seconds: float
    elapsed_seconds: float
    meta: dict[str, Any]


class VLMModel:
    """
    Default participant wrapper.

    `backend="dummy"` is for smoke tests only.
    `backend="transformers"` uses a local Hugging Face model directory.
    Participants can replace the internals while preserving
    `generate_with_metrics`.
    """

    def __init__(
        self,
        model_path: str,
        *,
        backend: str = "auto",
        device: str = "auto",
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.backend = backend
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._backend_name = "dummy"
        self._kivi_config = None

        if backend in {"auto", "transformers"}:
            try:
                self._load_transformers_backend()
                self._backend_name = "transformers"
            except Exception as exc:
                if backend == "transformers":
                    raise
                self._load_dummy_backend(str(exc))
        else:
            self._load_dummy_backend("backend=dummy")

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def generate_with_metrics(
        self,
        *,
        image,
        prompt: str,
        choices: dict[str, str],
        generation_config: GenerationConfig,
        sample_id: str,
    ) -> GenerationResult:
        if self._backend_name == "transformers":
            return self._generate_with_transformers(
                image=image,
                prompt=prompt,
                generation_config=generation_config,
            )
        return self._generate_with_dummy(
            prompt=prompt,
            choices=choices,
            generation_config=generation_config,
            sample_id=sample_id,
        )

    def _load_transformers_backend(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # Keep Windows console progress output readable by forcing ASCII bars.
        os.environ.setdefault("TQDM_ASCII", "1")
        self._install_transformers_log_filters()
        normalized_device = (self.device or "auto").strip().lower()
        dtype = torch.float32 if normalized_device == "cpu" else torch.bfloat16
        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        model_kwargs = {
            "local_files_only": True,
            "trust_remote_code": True,
            "dtype": dtype,
        }
        # Keep the baseline CPU path usable without forcing accelerate/device_map.
        if normalized_device == "auto":
            model_kwargs["device_map"] = "auto"
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        if normalized_device not in {"", "auto"}:
            self._model = self._model.to(torch.device(normalized_device))
        self._model = self._model.eval()
        self._tokenizer = getattr(self._processor, "tokenizer", None)

    def _install_transformers_log_filters(self) -> None:
        class _MessageFilter(logging.Filter):
            def __init__(self, blocked_substrings: tuple[str, ...]) -> None:
                super().__init__()
                self.blocked_substrings = blocked_substrings

            def filter(self, record: logging.LogRecord) -> bool:
                message = record.getMessage()
                return not any(token in message for token in self.blocked_substrings)

        qwen_logger = logging.getLogger("transformers.models.qwen3_5.modeling_qwen3_5")
        if not any(getattr(f, "_optiz_qwen_fast_path", False) for f in qwen_logger.filters):
            fast_path_filter = _MessageFilter(
                ("The fast path is not available because one of the required library is not installed.",)
            )
            fast_path_filter._optiz_qwen_fast_path = True
            qwen_logger.addFilter(fast_path_filter)

    def _load_dummy_backend(self, reason: str) -> None:
        self._dummy_reason = reason

    def _generate_with_transformers(
        self,
        *,
        image,
        prompt: str,
        generation_config: GenerationConfig,
    ) -> GenerationResult:
        import torch
        from transformers import TextIteratorStreamer

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)
        input_len = inputs.input_ids.shape[1]
        streamer = TextIteratorStreamer(
            self._processor.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        generation_kwargs = {
            **inputs,
            "max_new_tokens": generation_config.max_new_tokens,
            "do_sample": generation_config.temperature > 0,
            "use_cache": True,
            "streamer": streamer,
        }
        kivi_cache = self._build_kivi_cache_if_enabled()
        if kivi_cache is not None:
            generation_kwargs["past_key_values"] = kivi_cache
        if generation_config.temperature > 0:
            generation_kwargs["temperature"] = generation_config.temperature
            generation_kwargs["top_p"] = generation_config.top_p

        output_holder: dict[str, Any] = {}

        def _run_generate() -> None:
            try:
                with torch.no_grad():
                    output_holder["output_ids"] = self._model.generate(**generation_kwargs)
            except BaseException as exc:
                output_holder["error"] = exc
                streamer.end()

        worker = threading.Thread(target=_run_generate, daemon=True)
        start = time.perf_counter()
        worker.start()

        first_chunk_at = None
        chunks: list[str] = []
        for chunk in streamer:
            now = time.perf_counter()
            if first_chunk_at is None and chunk:
                first_chunk_at = now
            chunks.append(chunk)
        worker.join()
        if "error" in output_holder:
            raise RuntimeError("Transformers generation failed inside the worker thread.") from output_holder["error"]
        end = time.perf_counter()

        output_ids = output_holder["output_ids"]
        generated_ids = output_ids[0][input_len:]
        text = "".join(chunks).strip()
        if not text:
            text = self._processor.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

        ttft = (first_chunk_at - start) if first_chunk_at is not None else (end - start)
        return GenerationResult(
            text=text,
            token_count=int(generated_ids.shape[0]),
            ttft_seconds=ttft,
            elapsed_seconds=end - start,
            meta={
                "backend": "transformers",
                "kivi_kv_cache": kivi_cache.report().__dict__ if kivi_cache is not None else None,
            },
        )

    def _build_kivi_cache_if_enabled(self):
        enabled = os.environ.get("OPTIZ_QWEN_KIVI_KV_CACHE", "").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return None
        from optiz_qwen.compression import KiviConfig, build_qwen35_kivi_cache

        kivi_config = KiviConfig(
            k_bits=int(os.environ.get("OPTIZ_QWEN_KIVI_K_BITS", "2")),
            v_bits=int(os.environ.get("OPTIZ_QWEN_KIVI_V_BITS", "2")),
            group_size=int(os.environ.get("OPTIZ_QWEN_KIVI_GROUP_SIZE", "32")),
            residual_length=int(os.environ.get("OPTIZ_QWEN_KIVI_RESIDUAL_LENGTH", "32")),
        )
        return build_qwen35_kivi_cache(self._model.config, kivi_config)

    def _generate_with_dummy(
        self,
        *,
        prompt: str,
        choices: dict[str, str],
        generation_config: GenerationConfig,
        sample_id: str,
    ) -> GenerationResult:
        start = time.perf_counter()
        usable_choices = [key for key, value in choices.items() if (value or "").strip()]
        picked = usable_choices[hash(sample_id) % len(usable_choices)] if usable_choices else "A"
        text = (
            f"Answer: {picked}\n"
            "Explanation: dummy backend selected a deterministic option for smoke testing."
        )
        token_count = max(1, min(generation_config.max_new_tokens, len(text.split())))
        end = time.perf_counter()
        return GenerationResult(
            text=text,
            token_count=token_count,
            ttft_seconds=max(end - start, 1e-4),
            elapsed_seconds=max(end - start, 2e-4),
            meta={
                "backend": "dummy",
                "reason": getattr(self, "_dummy_reason", "n/a"),
                "prompt_chars": len(prompt),
            },
        )
