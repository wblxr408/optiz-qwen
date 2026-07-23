from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


RUNNER_VERSION = "RERUN-v0.1.1"
EXPECTED_BENCHMARK_VERSION = "dndx_public_self_test_v1.1"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    model_path: Path
    enable_gdn_fastpath: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible DNDX v1.1 CUDA comparison matrix."
    )
    parser.add_argument(
        "--code-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--baseline-model-path", type=Path, required=True)
    parser.add_argument("--awq-model-path", type=Path)
    parser.add_argument("--gdn-overlay", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=["baseline", "gdn-fast", "awq", "awq-gdn"],
        default=["baseline", "gdn-fast", "awq", "awq-gdn"],
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Use 0 to run the complete selected dataset.",
    )
    parser.add_argument("--sample-strategy", choices=["sequential", "stratified"], default="sequential")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-samples", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def executable_path(path: Path) -> Path:
    # Resolving a venv launcher symlink can bypass pyvenv.cfg and load another environment.
    return Path(os.path.abspath(path.expanduser()))


def model_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    files = [item for item in resolved.rglob("*") if item.is_file()]
    metadata_hashes = {}
    for name in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "quantization_config.json",
    ):
        candidate = resolved / name
        if candidate.is_file():
            metadata_hashes[name] = sha256_file(candidate)
    return {
        "requested_path": str(path),
        "resolved_path": str(resolved),
        "file_count": len(files),
        "total_bytes": sum(item.stat().st_size for item in files),
        "metadata_sha256": metadata_hashes,
    }


def run_capture(command: list[str], *, cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        return {"available": False, "error": str(exc)}
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def git_identity(code_root: Path) -> dict[str, Any]:
    head = run_capture(["git", "rev-parse", "HEAD"], cwd=code_root)
    status = run_capture(["git", "status", "--porcelain"], cwd=code_root)
    return {
        "head": head.get("stdout") if head.get("returncode") == 0 else None,
        "status_porcelain": (
            status.get("stdout") if status.get("returncode") == 0 else None
        ),
        "head_probe": head,
        "status_probe": status,
    }


def package_versions(python: Path) -> dict[str, Any]:
    code = (
        "import importlib.metadata as m,json,platform,sys;"
        "names=['torch','transformers','triton','causal-conv1d','fla-core',"
        "'llmcompressor','compressed-tensors','sglang'];"
        "versions={};"
        "\nfor name in names:\n"
        "  try: versions[name]=m.version(name)\n"
        "  except m.PackageNotFoundError: versions[name]=None\n"
        "print(json.dumps({'python':sys.version,'executable':sys.executable,"
        "'platform':platform.platform(),'packages':versions}))"
    )
    probe = run_capture([str(python), "-c", code])
    if probe.get("returncode") == 0:
        return json.loads(probe["stdout"])
    return {"probe": probe}


def resolve_cases(args: argparse.Namespace) -> list[CaseSpec]:
    requested = list(dict.fromkeys(args.cases))
    needs_awq = any(name in {"awq", "awq-gdn"} for name in requested)
    needs_gdn = any(name in {"gdn-fast", "awq-gdn"} for name in requested)
    if needs_awq and args.awq_model_path is None:
        raise ValueError("--awq-model-path is required for AWQ cases.")
    if needs_gdn and args.gdn_overlay is None:
        raise ValueError("--gdn-overlay is required for GDN cases.")

    mapping = {
        "baseline": CaseSpec("baseline", args.baseline_model_path, False),
        "gdn-fast": CaseSpec("gdn-fast", args.baseline_model_path, True),
        "awq": CaseSpec("awq", args.awq_model_path, False),
        "awq-gdn": CaseSpec("awq-gdn", args.awq_model_path, True),
    }
    return [mapping[name] for name in requested]


def build_environment(
    *,
    code_root: Path,
    spec: CaseSpec,
    gdn_overlay: Path | None,
    seed: int,
    cuda_visible_devices: str,
) -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str((code_root / "src").resolve())]
    if spec.enable_gdn_fastpath:
        if gdn_overlay is None:
            raise ValueError("GDN overlay is missing.")
        python_paths.insert(0, str(gdn_overlay.resolve()))
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONHASHSEED"] = str(seed)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    return env


def validate_result(
    path: Path,
    *,
    expected_samples: int | None,
    max_new_tokens: int,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("benchmark_version") != EXPECTED_BENCHMARK_VERSION:
        raise ValueError(
            f"{path} uses {payload.get('benchmark_version')!r}, expected "
            f"{EXPECTED_BENCHMARK_VERSION!r}."
        )
    if expected_samples is not None and payload.get("sample_count") != expected_samples:
        raise ValueError(f"{path} has an unexpected sample count.")
    if payload.get("generation", {}).get("max_new_tokens") != max_new_tokens:
        raise ValueError(f"{path} has an unexpected max_new_tokens value.")
    if not payload.get("public_validation", {}).get("passed", False):
        raise ValueError(f"{path} failed public result validation.")
    return payload


def summarize_results(
    results: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, payloads in results.items():
        ttft = [float(item["performance"]["avg_ttft_ms"]) for item in payloads]
        throughput = [
            float(item["performance"]["avg_throughput_tokens_per_sec"])
            for item in payloads
        ]
        accuracy = [float(item["accuracy"]["score"]) for item in payloads]
        summary[name] = {
            "runs": len(payloads),
            "sample_count_each": [int(item["sample_count"]) for item in payloads],
            "ttft_ms_mean": statistics.mean(ttft),
            "ttft_ms_stdev": statistics.stdev(ttft) if len(ttft) > 1 else 0.0,
            "throughput_tokens_per_sec_mean": statistics.mean(throughput),
            "throughput_tokens_per_sec_stdev": (
                statistics.stdev(throughput) if len(throughput) > 1 else 0.0
            ),
            "accuracy_mean": statistics.mean(accuracy),
            "accuracy_each": accuracy,
        }

    baseline = summary.get("baseline")
    if baseline:
        for name, metrics in summary.items():
            metrics["relative_to_baseline"] = {
                "ttft_percent": (
                    metrics["ttft_ms_mean"] / baseline["ttft_ms_mean"] - 1.0
                )
                * 100.0,
                "throughput_percent": (
                    metrics["throughput_tokens_per_sec_mean"]
                    / baseline["throughput_tokens_per_sec_mean"]
                    - 1.0
                )
                * 100.0,
                "accuracy_percentage_points": (
                    metrics["accuracy_mean"] - baseline["accuracy_mean"]
                )
                * 100.0,
            }
    return summary


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.num_samples < 0:
        raise ValueError("--num-samples must be zero or positive.")
    if args.repeats <= 0 or args.warmup_samples < 0:
        raise ValueError("Repeats must be positive and warmup samples non-negative.")
    if args.max_new_tokens != 256:
        raise ValueError("DNDX v1.1 reruns must use --max-new-tokens 256.")

    code_root = args.code_root.resolve()
    output_root = args.output_root.resolve()
    python = executable_path(args.python)
    dataset_path = args.dataset_path.resolve()
    cases = resolve_cases(args)
    for path in [code_root, dataset_path, python, *(spec.model_path for spec in cases)]:
        if not path.exists():
            raise FileNotFoundError(path)
    if any(spec.enable_gdn_fastpath for spec in cases):
        if args.gdn_overlay is None or not args.gdn_overlay.resolve().is_dir():
            raise FileNotFoundError(args.gdn_overlay)

    identity = git_identity(code_root)
    if args.expected_commit and identity["head"] != args.expected_commit:
        raise RuntimeError(
            f"Expected commit {args.expected_commit}, found {identity['head']}."
        )
    if args.require_clean and identity["status_porcelain"]:
        raise RuntimeError("The benchmark worktree is not clean.")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "runner_version": RUNNER_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "code": identity,
        "dataset": {
            "path": str(dataset_path),
            "sha256": sha256_file(dataset_path),
        },
        "models": {
            str(spec.model_path.resolve()): model_identity(spec.model_path)
            for spec in cases
        },
        "python": package_versions(python),
        "gpu": run_capture(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
        "config": {
            "cases": [asdict(spec) | {"model_path": str(spec.model_path)} for spec in cases],
            "num_samples": args.num_samples or None,
            "sample_strategy": args.sample_strategy,
            "repeats": args.repeats,
            "warmup_samples": args.warmup_samples,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
            "device": args.device,
            "cuda_visible_devices": args.cuda_visible_devices,
            "gdn_overlay": str(args.gdn_overlay.resolve()) if args.gdn_overlay else None,
        },
    }
    write_json(output_root / "运行清单.json", manifest)

    results: dict[str, list[dict[str, Any]]] = {}
    expected_samples = args.num_samples or None
    for spec in cases:
        case_dir = output_root / spec.name
        case_dir.mkdir(parents=True, exist_ok=True)
        env = build_environment(
            code_root=code_root,
            spec=spec,
            gdn_overlay=args.gdn_overlay,
            seed=args.seed,
            cuda_visible_devices=args.cuda_visible_devices,
        )
        results[spec.name] = []
        for run_index in range(1, args.repeats + 1):
            result_path = case_dir / f"run{run_index}.json"
            log_path = case_dir / f"run{run_index}.log"
            if result_path.exists():
                if not args.resume:
                    raise FileExistsError(
                        f"{result_path} exists; use --resume or a new output root."
                    )
                payload = validate_result(
                    result_path,
                    expected_samples=expected_samples,
                    max_new_tokens=args.max_new_tokens,
                )
                results[spec.name].append(payload)
                print(f"[resume] {spec.name} run {run_index}", flush=True)
                continue

            command = [
                str(python),
                str(code_root / "benchmark_public.py"),
                "--dataset-path",
                str(dataset_path),
                "--model-path",
                str(spec.model_path.resolve()),
                "--output",
                str(result_path),
                "--sample-strategy",
                args.sample_strategy,
                "--seed",
                str(args.seed),
                "--warmup-samples",
                str(args.warmup_samples),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--backend",
                "transformers",
                "--device",
                args.device,
            ]
            if args.num_samples:
                command.extend(["--num-samples", str(args.num_samples)])

            print(f"[run] {spec.name} {run_index}/{args.repeats}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    command,
                    cwd=code_root,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"{spec.name} run {run_index} failed; see {log_path}."
                )
            payload = validate_result(
                result_path,
                expected_samples=expected_samples,
                max_new_tokens=args.max_new_tokens,
            )
            results[spec.name].append(payload)
            write_json(
                output_root / "汇总_进行中.json",
                {
                    "runner_version": RUNNER_VERSION,
                    "completed_at": datetime.now().astimezone().isoformat(),
                    "summary": summarize_results(results),
                },
            )

    final_payload = {
        "runner_version": RUNNER_VERSION,
        "completed_at": datetime.now().astimezone().isoformat(),
        "benchmark_version": EXPECTED_BENCHMARK_VERSION,
        "summary": summarize_results(results),
    }
    write_json(output_root / "汇总.json", final_payload)
    print(json.dumps(final_payload, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
