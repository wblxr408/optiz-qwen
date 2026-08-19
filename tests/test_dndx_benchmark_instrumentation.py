from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace


SRC_DIR = Path(__file__).resolve().parents[1] / "src/optiz_qwen"
EVALUATION_DIR = SRC_DIR / "evaluation"
TEST_PACKAGE = "_dndx_benchmark_instrumentation_test_package"

package = ModuleType(TEST_PACKAGE)
package.__path__ = [str(SRC_DIR)]
sys.modules[TEST_PACKAGE] = package

evaluation_package_name = f"{TEST_PACKAGE}.evaluation"
evaluation_package = ModuleType(evaluation_package_name)
evaluation_package.__path__ = [str(EVALUATION_DIR)]
sys.modules[evaluation_package_name] = evaluation_package

answer_parsing = ModuleType(f"{evaluation_package_name}.answer_parsing")
answer_parsing.extract_answer = lambda text: None
answer_parsing.parse_choice_answer = lambda text, choices: (None, "test")
sys.modules[answer_parsing.__name__] = answer_parsing

dndx_wrapper = ModuleType(f"{evaluation_package_name}.dndx_wrapper")
dndx_wrapper.GenerationConfig = SimpleNamespace
dndx_wrapper.VLMModel = object
sys.modules[dndx_wrapper.__name__] = dndx_wrapper

scheduling = ModuleType(f"{TEST_PACKAGE}.scheduling")
scheduling.prefill_last_logit_only_enabled = lambda: True
sys.modules[scheduling.__name__] = scheduling

MODULE_NAME = f"{evaluation_package_name}.dndx_public_benchmark"
SPEC = importlib.util.spec_from_file_location(
    MODULE_NAME,
    EVALUATION_DIR / "dndx_public_benchmark.py",
)
assert SPEC is not None and SPEC.loader is not None
dndx_public_benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = dndx_public_benchmark
SPEC.loader.exec_module(dndx_public_benchmark)


class BenchmarkInstrumentationTests(unittest.TestCase):
    def test_latency_percentiles_use_linear_interpolation(self) -> None:
        metrics = dndx_public_benchmark._build_runtime_metrics(
            [0.010, 0.020, 0.030, 0.040],
            [1, 1, 1, 1],
            processor_load_time_sec=None,
            model_load_time_sec=None,
        )

        self.assertEqual(metrics["latency_mean_ms"], 25.0)
        self.assertEqual(metrics["latency_p50_ms"], 25.0)
        self.assertEqual(metrics["latency_p95_ms"], 38.5)

    def test_aggregate_output_throughput_uses_formal_totals(self) -> None:
        metrics = dndx_public_benchmark._build_runtime_metrics(
            [1.0, 2.0],
            [10, 20],
            processor_load_time_sec=0.5,
            model_load_time_sec=1.5,
        )

        self.assertEqual(metrics["total_output_tokens"], 30)
        self.assertEqual(metrics["total_generation_time_sec"], 3.0)
        self.assertEqual(metrics["aggregate_output_tokens_per_sec"], 10.0)

    def test_warmup_values_are_excluded_from_formal_aggregation(self) -> None:
        warmup_latencies = [99.0]
        warmup_tokens = [999]
        formal_latencies = [1.0, 2.0]
        formal_tokens = [10, 20]

        metrics = dndx_public_benchmark._build_runtime_metrics(
            formal_latencies,
            formal_tokens,
            processor_load_time_sec=None,
            model_load_time_sec=None,
        )

        self.assertNotEqual(
            metrics["total_generation_time_sec"],
            sum(warmup_latencies + formal_latencies),
        )
        self.assertNotEqual(
            metrics["total_output_tokens"],
            sum(warmup_tokens + formal_tokens),
        )
        self.assertEqual(metrics["total_generation_time_sec"], 3.0)
        self.assertEqual(metrics["total_output_tokens"], 30)

    def test_unavailable_memory_metrics_return_none_without_error(self) -> None:
        cpu_cuda = SimpleNamespace(
            memory_allocated=lambda device: self.fail("CPU memory API was called")
        )
        cpu_model = SimpleNamespace(
            _torch=SimpleNamespace(cuda=cpu_cuda),
            _resolved_device="cpu",
        )
        cuda_model_without_memory_apis = SimpleNamespace(
            _torch=SimpleNamespace(cuda=SimpleNamespace()),
            _resolved_device="cuda:0",
        )

        self.assertIsNone(
            dndx_public_benchmark._cuda_memory_gb(
                cpu_model, "memory_allocated"
            )
        )
        self.assertIsNone(
            dndx_public_benchmark._cuda_memory_gb(
                cuda_model_without_memory_apis, "memory_allocated"
            )
        )
        self.assertFalse(
            dndx_public_benchmark._reset_cuda_peak_memory_stats(
                cuda_model_without_memory_apis
            )
        )


if __name__ == "__main__":
    unittest.main()
