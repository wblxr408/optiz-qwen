from __future__ import annotations

from argparse import Namespace

from optiz_qwen.evaluation.answer_parsing import parse_choice_answer
from evaluation_wrapper import GenerationConfig, GenerationResult, VLMModel
from optiz_qwen.evaluation.dndx_public_benchmark import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_PATH,
    compute_throughput,
    extract_answer,
    kivi_cli_environment,
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
    assert extract_answer("no valid answer") is None
    assert round(compute_throughput(5, 0.2, 1.0), 3) == 5.0


def test_choice_text_answer_inference() -> None:
    answer, source = parse_choice_answer(
        "这项实验能回答：当麦德琳的雪板上有一层蜡或没有蜡时，它是否能在较短的时间内滑下山坡？",
        {
            "A": "当麦德琳的雪板上有一层薄蜡或一层厚蜡时，它是否能在较短的时间内滑下山坡？",
            "B": "当麦德琳的雪板上有一层蜡或没有蜡时，它是否能在较短的时间内滑下山坡？",
            "C": "",
            "D": "",
        },
    )

    assert answer == "B"
    assert source == "exact_choice_text"


def test_default_paths_match_repo_layout() -> None:
    assert DEFAULT_DATASET_PATH == "./resources/eval_dataset/raw/mmbench_public/mmbench_dev_en.tsv"
    assert DEFAULT_MODEL_PATH == "./resources/model_weights/raw/Qwen3.5-2B"
    assert DEFAULT_OUTPUT_PATH == "./benchmarks/output/result_public.json"


def test_kivi_cli_environment_sets_and_restores_env(monkeypatch) -> None:
    monkeypatch.delenv("OPTIZ_QWEN_KIVI_KV_CACHE", raising=False)
    args = Namespace(
        enable_kivi_kv_cache=True,
        kivi_k_bits=2,
        kivi_v_bits=4,
        kivi_group_size=32,
        kivi_residual_length=64,
    )

    with kivi_cli_environment(args):
        import os

        assert os.environ["OPTIZ_QWEN_KIVI_KV_CACHE"] == "1"
        assert os.environ["OPTIZ_QWEN_KIVI_V_BITS"] == "4"
        assert os.environ["OPTIZ_QWEN_KIVI_RESIDUAL_LENGTH"] == "64"

    import os

    assert "OPTIZ_QWEN_KIVI_KV_CACHE" not in os.environ
