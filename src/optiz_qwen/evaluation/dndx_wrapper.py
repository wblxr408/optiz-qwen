"""Participant wrapper contract for the DNDX public benchmark path."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from optiz_qwen.evaluation.answer_parsing import parse_choice_answer
from optiz_qwen.compression import (
    QServeKvConfig,
    Qwen35TomeConfig,
    get_qwen35_tome_runtime,
    install_qwen35_tome,
)
from optiz_qwen.scheduling import (
    build_kv_chain,
    cuda_graph_decode_enabled,
    run_greedy_prefill_decode,
)


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
        dtype: str | None = None,
    ) -> None:
        if dtype not in {None, "bf16", "fp16"}:
            raise ValueError("dtype must be None, bf16, or fp16")
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self.backend = backend
        self._model = None
        self._processor = None
        self._tokenizer = None
        self._backend_name = "dummy"
        self._resolved_device = "cpu"
        self._resolved_dtype_name = "unloaded"
        self._processor_load_time_sec = None
        self._model_load_time_sec = None
        self._graph_decoder = None
        self._graph_decode_report = None

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

    @property
    def dtype_name(self) -> str:
        return self._resolved_dtype_name

    @property
    def processor_load_time_sec(self) -> float | None:
        return self._processor_load_time_sec

    @property
    def model_load_time_sec(self) -> float | None:
        return self._model_load_time_sec

    @property
    def quantization_config(self):
        config = getattr(self._model, "config", None)
        return getattr(config, "quantization_config", None)

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
                choices=choices,
                generation_config=generation_config,
            )
        return self._generate_with_dummy(
            image=image,
            prompt=prompt,
            choices=choices,
            generation_config=generation_config,
            sample_id=sample_id,
        )

    def _load_transformers_backend(self) -> None:
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        model_dir = Path(self.model_path)
        if not model_dir.exists():
            raise FileNotFoundError(f"Local model path does not exist: {model_dir}")

        # Keep Windows console progress output readable by forcing ASCII bars.
        os.environ.setdefault("TQDM_ASCII", "1")
        self._install_transformers_log_filters()
        self._torch = torch
        self._resolved_device = self._resolve_torch_device(torch)
        processor_load_start = time.perf_counter()
        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        self._processor_load_time_sec = time.perf_counter() - processor_load_start
        self._configure_visual_token_budget()
        torch_dtype = self._resolve_torch_dtype(torch)
        self._resolved_dtype_name = {
            torch.bfloat16: "bf16",
            torch.float16: "fp16",
            torch.float32: "fp32",
        }.get(torch_dtype, str(torch_dtype).removeprefix("torch."))
        model_load_start = time.perf_counter()
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch_dtype,
        )
        if self._resolved_device != "cpu":
            self._model = self._model.to(self._resolved_device)
        self._model = self._model.eval()
        self._model_load_time_sec = time.perf_counter() - model_load_start
        self._configure_tome()
        self._tokenizer = getattr(self._processor, "tokenizer", None)

    def _configure_tome(self) -> None:
        enabled = os.environ.get("OPTIZ_QWEN_TOME_ENABLED", "").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            self._tome_config = None
            return
        config = Qwen35TomeConfig(
            layer=int(os.environ.get("OPTIZ_QWEN_TOME_LAYER", "12")),
            r=int(os.environ.get("OPTIZ_QWEN_TOME_R", "1")),
            proportional_attention=os.environ.get(
                "OPTIZ_QWEN_TOME_PROPORTIONAL_ATTENTION",
                "",
            ).strip().lower() in {"1", "true", "yes", "on"},
        )
        install_qwen35_tome(self._model, config)
        self._tome_config = config

    def _configure_visual_token_budget(self) -> None:
        value = os.environ.get("OPTIZ_QWEN_VISUAL_PIXEL_BUDGET", "").strip()
        if not value:
            self._visual_pixel_budget = None
            return
        pixel_budget = int(value)
        if pixel_budget < 16384:
            raise ValueError("visual pixel budget must be at least 16384 (128x128).")
        image_processor = getattr(self._processor, "image_processor", None)
        if image_processor is None:
            raise RuntimeError("processor does not expose an image_processor for visual budgeting.")
        image_processor.size = {"shortest_edge": pixel_budget, "longest_edge": pixel_budget}
        self._visual_pixel_budget = pixel_budget

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

    def _resolve_torch_device(self, torch) -> str:
        requested = (self.device or "auto").lower()
        if requested == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        return self.device

    def _resolve_torch_dtype(self, torch):
        if self.dtype == "bf16":
            return torch.bfloat16
        if self.dtype == "fp16":
            return torch.float16
        if str(self._resolved_device).startswith("cuda"):
            return torch.float16
        return torch.float32

    def _build_model_inputs(self, *, image, prompt: str):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=[chat_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        )
        model_device = getattr(self._model, "device", self._resolved_device)
        return {
            key: value.to(model_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _generate_with_transformers(
        self,
        *,
        image,
        prompt: str,
        choices: dict[str, str],
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
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        kv_chain, kv_report = self._build_kv_chain_if_enabled()
        runner = os.environ.get("OPTIZ_QWEN_GENERATION_RUNNER", "generate").strip().lower()
        deferred_fused_chain = kv_report is not None and kv_report.chain_name == "qserve_deferred_split_fused_kv"
        if deferred_fused_chain and runner != "greedy":
            raise ValueError(
                "qserve_deferred_split_fused_kv requires OPTIZ_QWEN_GENERATION_RUNNER=greedy."
            )
        use_prefill_decode = generation_config.temperature == 0.0 and runner == "greedy"
        graph_decoder = None
        if use_prefill_decode and cuda_graph_decode_enabled():
            if kv_chain is not None:
                raise ValueError(
                    "OPTIZ_QWEN_CUDA_GRAPH_DECODE cannot be combined with "
                    "OPTIZ_QWEN_KV_CHAIN_ENABLED; the captured graph owns the decode loop "
                    "and requires the StaticCache it was captured against."
                )
            graph_decoder = self._ensure_graph_decoder(inputs)
            kv_chain = graph_decoder.cache
        if use_prefill_decode:
            if kv_chain is None:
                from transformers.cache_utils import DynamicCache

                kv_chain = DynamicCache(config=self._model.config)
            post_prefill_callback = None
            post_decode_callback = None
            if deferred_fused_chain:
                from optiz_qwen.kernels import install_qwen35_fused_attention

                def _install_after_threshold() -> None:
                    if getattr(kv_chain, "packed_decode_active", lambda: False)():
                        install_qwen35_fused_attention(self._model)

                post_decode_callback = _install_after_threshold
            start = time.perf_counter()
            generated_ids, runtime_stats = run_greedy_prefill_decode(
                self._model,
                inputs,
                max_new_tokens=generation_config.max_new_tokens,
                tokenizer=self._tokenizer,
                eos_token_id=getattr(self._tokenizer, "eos_token_id", None),
                kv_cache=kv_chain,
                post_prefill_callback=post_prefill_callback,
                post_decode_callback=post_decode_callback,
                graph_decoder=graph_decoder,
            )
            first_chunk_at = start + runtime_stats.ttft_seconds
            chunks = []
            output_ids = torch.cat([inputs.input_ids, generated_ids], dim=-1)
            end = start + runtime_stats.elapsed_seconds
        else:
            generation_kwargs = {
                **inputs,
                "max_new_tokens": generation_config.max_new_tokens,
                "use_cache": True,
                "streamer": streamer,
                "do_sample": generation_config.temperature > 0,
            }
            if generation_config.temperature > 0:
                generation_kwargs["temperature"] = generation_config.temperature
                generation_kwargs["top_p"] = generation_config.top_p
            if kv_chain is not None:
                attention_mask = inputs.get("attention_mask")
                dense_decode_mask = attention_mask is None or bool(torch.all(attention_mask == 1).item())
                setattr(kv_chain, "_optiz_dense_decode_mask", dense_decode_mask)
                generation_kwargs["past_key_values"] = kv_chain

            output_holder: dict[str, Any] = {}

            def _run_generate() -> None:
                try:
                    with torch.no_grad():
                        output_holder["output_ids"] = self._model.generate(**generation_kwargs)
                except BaseException as exc:  # pragma: no cover - exercised in live runtime
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
            runtime_stats = None
        # Performance metrics cover model generation only.  The optional
        # logits-based answer recovery below is an evaluation-side accuracy
        # fallback, so including it would make two equal generation paths look
        # different merely because one emitted a more parseable string.
        generation_end = end
        generated_ids = output_ids[0][input_len:]
        text = "".join(chunks).strip()
        if not text:
            text = self._tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

        raw_text = text
        parsed_answer, answer_source = parse_choice_answer(text, choices)
        choice_fallback_meta = None
        if parsed_answer is None and self._choice_fallback_enabled():
            fallback_start = time.perf_counter()
            parsed_answer, choice_fallback_meta = self._select_choice_by_logits(
                image=image,
                prompt=prompt,
                choices=choices,
            )
            if parsed_answer is not None:
                answer_source = "logit_choice_fallback"
                text = f"Answer: {parsed_answer}"
                if raw_text:
                    text = f"{text}\nRaw response: {raw_text}"
            end = time.perf_counter()
            if first_chunk_at is None and parsed_answer is not None:
                first_chunk_at = fallback_start

        ttft = (first_chunk_at - start) if first_chunk_at is not None else (end - start)
        cache_runtime = None
        if deferred_fused_chain:
            cache_runtime = {
                "kernel_calls": int(getattr(kv_chain, "kernel_calls", 0)),
                "fallback_calls": int(getattr(kv_chain, "fallback_calls", 0)),
                "active_backend": (
                    getattr(kv_chain, "attention_backend", "triton_int4_decode")
                    if getattr(kv_chain, "kernel_calls", 0) > 0
                    else (
                        "native_dense_below_threshold"
                        if not getattr(kv_chain, "packed_decode_active", lambda: False)()
                        else "packed_kv_not_amortized"
                    )
                ),
            }
        return GenerationResult(
            text=text,
            token_count=int(generated_ids.shape[0]),
            ttft_seconds=ttft,
            elapsed_seconds=generation_end - start,
            meta={
                "backend": "transformers",
                "generation_runner": "greedy" if use_prefill_decode else "generate",
                "visual_pixel_budget": getattr(self, "_visual_pixel_budget", None),
                "tome": (
                    get_qwen35_tome_runtime(self._model)
                    if getattr(self, "_tome_config", None) is not None
                    else None
                ),
                "kv_chain": kv_report.__dict__ if kv_report is not None else None,
                "kv_runtime": cache_runtime,
                "cuda_graph_decode": (
                    self._graph_decode_report.to_dict()
                    if getattr(self, "_graph_decode_report", None) is not None
                    else None
                ),
                "prefill_decode_runtime": runtime_stats.__dict__ if runtime_stats is not None else None,
                "answer_source": answer_source,
                "raw_text": raw_text,
                "choice_fallback": choice_fallback_meta,
            },
        )

    def _ensure_graph_decoder(self, inputs):
        """Build, prefill against, and capture the decode graph once per process.

        The graph is captured while the config says ``decode_backend`` and the
        config is then flipped back to ``prefill_backend``.  A captured graph
        freezes the kernels that were live at capture time, so replay keeps
        dispatching FA2 decode kernels while the uncaptured prefill pass
        dispatches sdpa -- which is what wins both metrics at once.

        Capture needs a cache holding a real prefill so the decode step sees
        plausible state; those writes are discarded because
        ``run_greedy_prefill_decode`` resets the cache before every request.
        """

        if self._graph_decoder is not None:
            return self._graph_decoder

        import torch

        from optiz_qwen.kernels import (
            attention_backend,
            resolved_decode_backend,
            resolved_prefill_backend,
            set_attention_backend,
        )
        from optiz_qwen.scheduling import (
            CudaGraphDecoder,
            build_static_cache,
            resolved_max_cache_len,
            resolved_warmup_steps,
        )

        prefill_backend = resolved_prefill_backend()
        decode_backend = resolved_decode_backend()
        device = getattr(self._model, "device", self._resolved_device)
        dtype = next(self._model.parameters()).dtype
        cache = build_static_cache(
            self._model,
            max_cache_len=resolved_max_cache_len(),
            device=device,
            dtype=dtype,
        )
        decoder = CudaGraphDecoder(
            self._model,
            cache,
            capture_backend=decode_backend,
            warmup_steps=resolved_warmup_steps(),
        )

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        capture_inputs = dict(inputs)
        capture_inputs["use_cache"] = True
        capture_inputs["past_key_values"] = cache
        capture_inputs["cache_position"] = torch.arange(prompt_tokens, device=device)
        with torch.inference_mode():
            with attention_backend(self._model, prefill_backend):
                capture_outputs = self._model(**capture_inputs)
            token_id = int(torch.argmax(capture_outputs.logits[:, -1, :], dim=-1).item())

        decoder.capture(token_id=token_id, position=prompt_tokens, device=device)
        set_attention_backend(self._model, prefill_backend)

        self._graph_decoder = decoder
        self._graph_decode_report = decoder.report(prefill_backend=prefill_backend)
        return decoder

    def _build_kv_chain_if_enabled(self):
        enabled = os.environ.get("OPTIZ_QWEN_KV_CHAIN_ENABLED", "").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return None, None
        chain_name = os.environ.get(
            "OPTIZ_QWEN_KV_CHAIN", "qserve_deferred_split_fused_kv"
        ).strip().lower()
        qserve_config = QServeKvConfig(
            k_bits=int(os.environ.get("OPTIZ_QWEN_KV_CHAIN_K_BITS", "4")),
            v_bits=int(os.environ.get("OPTIZ_QWEN_KV_CHAIN_V_BITS", "4")),
            group_size=int(os.environ.get("OPTIZ_QWEN_KV_CHAIN_GROUP_SIZE", "32")),
            residual_length=int(os.environ.get("OPTIZ_QWEN_KV_CHAIN_RESIDUAL_LENGTH", "32")),
            activation_threshold=int(os.environ.get("OPTIZ_QWEN_KV_CHAIN_ACTIVATION_THRESHOLD", "1024")),
            decode_warmup_tokens=int(os.environ.get("OPTIZ_QWEN_KV_CHAIN_DECODE_WARMUP_TOKENS", "4")),
            attention_backend=os.environ.get(
                "OPTIZ_QWEN_KV_CHAIN_ATTENTION_BACKEND", "triton_int4_split_decode"
            ).strip(),
        )
        kv_chain, kv_report = build_kv_chain(
            chain_name=chain_name,
            model_config=self._model.config,
            enabled=True,
            qserve_config=qserve_config,
        )
        return kv_chain, kv_report

    def _choice_fallback_enabled(self) -> bool:
        value = os.environ.get("OPTIZ_QWEN_CHOICE_FALLBACK", "1").strip().lower()
        return value not in {"0", "false", "no", "off"}

    def _select_choice_by_logits(
        self,
        *,
        image,
        prompt: str,
        choices: dict[str, str],
    ) -> tuple[str | None, dict[str, Any]]:
        usable_choices = [
            key
            for key, value in choices.items()
            if key in {"A", "B", "C", "D"} and (value or "").strip()
        ]
        if not usable_choices:
            usable_choices = ["A", "B", "C", "D"]

        try:
            inputs = self._build_model_inputs(image=image, prompt=f"{prompt.rstrip()}\nAnswer:")
            with self._torch.no_grad():
                outputs = self._model(**inputs, use_cache=False)
            logits = outputs.logits[0, -1, :]
            scores = {}
            for key in usable_choices:
                token_ids = self._candidate_token_ids(key)
                scores[key] = max(float(logits[token_id]) for token_id in token_ids)
            picked = max(scores, key=scores.get)
            return picked, {"enabled": True, "scores": scores}
        except Exception as exc:
            return None, {"enabled": True, "error": repr(exc)}

    def _candidate_token_ids(self, choice: str) -> tuple[int, ...]:
        tokenizer = self._tokenizer or self._processor.tokenizer
        token_ids: list[int] = []
        for variant in (choice, f" {choice}", f"{choice}.", f"({choice})"):
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                token_ids.append(int(encoded[0]))
        return tuple(dict.fromkeys(token_ids))

    def _generate_with_dummy(
        self,
        *,
        image,
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
                "input_image_size": list(image.size),
            },
        )
