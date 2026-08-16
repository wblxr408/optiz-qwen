from __future__ import annotations

from argparse import Namespace

import pytest
from optiz_qwen.evaluation.answer_parsing import parse_choice_answer
from evaluation_wrapper import GenerationConfig, GenerationResult, VLMModel
from optiz_qwen.evaluation.dndx_public_benchmark import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_PATH,
    HYBRID_ENV_KEYS,
    OFFICIAL_MAX_NEW_TOKENS,
    compute_throughput,
    extract_answer,
    fixed_generation_config,
    hybrid_cli_environment,
    kv_chain_cli_environment,
    run_benchmark,
    Sample,
    select_samples,
    runner_cli_environment,
    tome_cli_environment,
    visual_cli_environment,
)


def test_wrapper_contract_exports() -> None:
    config = GenerationConfig(max_new_tokens=8)
    result = GenerationResult(
        text="Answer: A",
        token_count=3,
        ttft_seconds=0.01,
        elapsed_seconds=0.02,
        meta={"backend": "dummy"},
    )
    model = VLMModel("unused", backend="dummy")

    assert config.max_new_tokens == 8
    assert result.token_count == 3
    assert model.backend_name == "dummy"


def test_answer_extraction_and_metrics() -> None:
    assert extract_answer("Answer: B") == "B"
    assert extract_answer("答案：C") == "C"
    assert extract_answer("正确答案是（D）。") == "D"
    assert extract_answer("我选 A") == "A"
    assert extract_answer("应该选【B】") == "B"
    assert extract_answer("Final choice is (C)") == "C"
    assert extract_answer("no valid answer") is None
    assert round(compute_throughput(5, 0.2, 1.0), 3) == 5.0


def test_choice_text_is_not_inferred_outside_official_rules() -> None:
    answer, source = parse_choice_answer(
        "这项实验能回答：当麦德琳的雪板上有一层蜡或没有蜡时，它是否能在较短的时间内滑下山坡？",
        {
            "A": "当麦德琳的雪板上有一层薄蜡或一层厚蜡时，它是否能在较短的时间内滑下山坡？",
            "B": "当麦德琳的雪板上有一层蜡或没有蜡时，它是否能在较短的时间内滑下山坡？",
            "C": "",
            "D": "",
        },
    )

    assert answer is None
    assert source == "missing_choice_answer"


def test_default_paths_match_repo_layout() -> None:
    assert DEFAULT_DATASET_PATH == "./resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv"
    assert DEFAULT_MODEL_PATH == "./resources/model_weights/raw/Qwen3.5-2B"
    assert DEFAULT_OUTPUT_PATH == "./benchmarks/output/result_public.json"
    assert OFFICIAL_MAX_NEW_TOKENS == 256
    assert fixed_generation_config().max_new_tokens == 256


def test_kv_chain_cli_environment_sets_and_restores_env(monkeypatch) -> None:
    monkeypatch.delenv("OPTIZ_QWEN_KV_CHAIN_ENABLED", raising=False)
    monkeypatch.delenv("OPTIZ_QWEN_KV_CHAIN", raising=False)
    args = Namespace(
        enable_kv_chain=True,
        kv_chain="qserve_deferred_split_fused_kv",
        kv_chain_k_bits=4,
        kv_chain_v_bits=4,
        kv_chain_group_size=32,
        kv_chain_residual_length=32,
        kv_chain_activation_threshold=1024,
        kv_chain_decode_warmup_tokens=4,
        kv_chain_attention_backend="triton_int4_split_decode",
    )

    with kv_chain_cli_environment(args):
        import os

        assert os.environ["OPTIZ_QWEN_KV_CHAIN_ENABLED"] == "1"
        assert os.environ["OPTIZ_QWEN_KV_CHAIN"] == "qserve_deferred_split_fused_kv"
        assert os.environ["OPTIZ_QWEN_KV_CHAIN_ACTIVATION_THRESHOLD"] == "1024"
        assert os.environ["OPTIZ_QWEN_KV_CHAIN_DECODE_WARMUP_TOKENS"] == "4"
        assert os.environ["OPTIZ_QWEN_KV_CHAIN_ATTENTION_BACKEND"] == "triton_int4_split_decode"

    import os

    assert "OPTIZ_QWEN_KV_CHAIN_ENABLED" not in os.environ
    assert "OPTIZ_QWEN_KV_CHAIN" not in os.environ


def test_kv_chain_cli_environment_disables_inherited_setting(monkeypatch) -> None:
    monkeypatch.setenv("OPTIZ_QWEN_KV_CHAIN_ENABLED", "1")
    monkeypatch.setenv("OPTIZ_QWEN_KV_CHAIN", "qserve_deferred_split_fused_kv")
    args = Namespace(enable_kv_chain=False)

    with kv_chain_cli_environment(args):
        import os

        assert "OPTIZ_QWEN_KV_CHAIN_ENABLED" not in os.environ
        assert "OPTIZ_QWEN_KV_CHAIN" not in os.environ

    assert os.environ["OPTIZ_QWEN_KV_CHAIN_ENABLED"] == "1"
    assert os.environ["OPTIZ_QWEN_KV_CHAIN"] == "qserve_deferred_split_fused_kv"


def test_kv_chain_cli_environment_rejects_removed_route() -> None:
    args = Namespace(enable_kv_chain=True, kv_chain="qserve_kv")

    with pytest.raises(ValueError, match="Only qserve_deferred_split_fused_kv is retained"):
        with kv_chain_cli_environment(args):
            pass


def test_runner_and_visual_environments_are_scoped(monkeypatch) -> None:
    import os

    monkeypatch.delenv("OPTIZ_QWEN_GENERATION_RUNNER", raising=False)
    monkeypatch.delenv("OPTIZ_QWEN_VISUAL_PIXEL_BUDGET", raising=False)
    args = Namespace(generation_runner="greedy", visual_pixel_budget=36864)

    with runner_cli_environment(args), visual_cli_environment(args):
        assert os.environ["OPTIZ_QWEN_GENERATION_RUNNER"] == "greedy"
        assert os.environ["OPTIZ_QWEN_VISUAL_PIXEL_BUDGET"] == "36864"

    assert "OPTIZ_QWEN_GENERATION_RUNNER" not in os.environ
    assert "OPTIZ_QWEN_VISUAL_PIXEL_BUDGET" not in os.environ


def test_tome_cli_environment_is_explicit_and_scoped(monkeypatch) -> None:
    import os

    monkeypatch.setenv("OPTIZ_QWEN_TOME_ENABLED", "1")
    disabled_args = Namespace(enable_tome=False)
    with tome_cli_environment(disabled_args):
        assert "OPTIZ_QWEN_TOME_ENABLED" not in os.environ
    assert os.environ["OPTIZ_QWEN_TOME_ENABLED"] == "1"

    enabled_args = Namespace(
        enable_tome=True,
        tome_layer=8,
        tome_r=2,
        tome_proportional_attention=True,
    )
    with tome_cli_environment(enabled_args):
        assert os.environ["OPTIZ_QWEN_TOME_ENABLED"] == "1"
        assert os.environ["OPTIZ_QWEN_TOME_LAYER"] == "8"
        assert os.environ["OPTIZ_QWEN_TOME_R"] == "2"
        assert os.environ["OPTIZ_QWEN_TOME_PROPORTIONAL_ATTENTION"] == "1"


def test_hybrid_cli_environment_sets_and_restores_env(monkeypatch) -> None:
    import os

    for key in HYBRID_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    args = Namespace(
        enable_hybrid_cudagraph=True,
        hybrid_max_cache_len=4096,
        hybrid_warmup_steps=2,
        hybrid_prefill_backend="sdpa",
        hybrid_decode_backend="flash_attention_2",
    )

    with hybrid_cli_environment(args):
        assert os.environ["OPTIZ_QWEN_CUDA_GRAPH_DECODE"] == "1"
        assert os.environ["OPTIZ_QWEN_CUDA_GRAPH_MAX_CACHE_LEN"] == "4096"
        assert os.environ["OPTIZ_QWEN_CUDA_GRAPH_WARMUP_STEPS"] == "2"
        # The split is the whole point: capture under FA2, run prefill on sdpa.
        assert os.environ["OPTIZ_QWEN_ATTN_PREFILL"] == "sdpa"
        assert os.environ["OPTIZ_QWEN_ATTN_DECODE"] == "flash_attention_2"

    for key in HYBRID_ENV_KEYS:
        assert key not in os.environ


def test_hybrid_cli_environment_disables_inherited_setting(monkeypatch) -> None:
    import os

    monkeypatch.setenv("OPTIZ_QWEN_CUDA_GRAPH_DECODE", "1")
    monkeypatch.setenv("OPTIZ_QWEN_ATTN_DECODE", "flash_attention_2")

    with hybrid_cli_environment(Namespace(enable_hybrid_cudagraph=False)):
        assert "OPTIZ_QWEN_CUDA_GRAPH_DECODE" not in os.environ
        assert "OPTIZ_QWEN_ATTN_DECODE" not in os.environ

    assert os.environ["OPTIZ_QWEN_CUDA_GRAPH_DECODE"] == "1"
    assert os.environ["OPTIZ_QWEN_ATTN_DECODE"] == "flash_attention_2"


def test_hybrid_requires_the_greedy_runner() -> None:
    args = Namespace(
        enable_hybrid_cudagraph=True,
        enable_kv_chain=False,
        generation_runner="generate",
    )

    with pytest.raises(ValueError, match="requires --generation-runner greedy"):
        run_benchmark(args)


def test_hybrid_is_mutually_exclusive_with_the_kv_chain() -> None:
    args = Namespace(
        enable_hybrid_cudagraph=True,
        enable_kv_chain=True,
        generation_runner="greedy",
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        run_benchmark(args)


def test_stratified_sample_selection_is_balanced_and_reproducible() -> None:
    samples = [
        Sample(str(index), "cn", "q", "", {"A": "a"}, "A", "", category, "sub")
        for category in ("ocr", "object_localization")
        for index in range(5)
    ]

    first = select_samples(
        samples,
        limit=6,
        strategy="stratified",
        seed=17,
        categories={"ocr", "object_localization"},
    )
    second = select_samples(
        samples,
        limit=6,
        strategy="stratified",
        seed=17,
        categories={"ocr", "object_localization"},
    )

    assert [sample.sample_id for sample in first] == [sample.sample_id for sample in second]
    assert [sample.category for sample in first].count("ocr") == 3
    assert [sample.category for sample in first].count("object_localization") == 3
