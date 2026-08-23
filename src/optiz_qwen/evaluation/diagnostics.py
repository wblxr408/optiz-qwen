"""Crash diagnostics for the public benchmark on native-abort hardware.

Why this module exists
----------------------
On PPU-ZW810E a smoke run can terminate with::

    bash: line 5: 214248 Aborted
    rc=134

``rc=134`` is ``128 + 6`` -- ``SIGABRT`` -- raised by the native
CUDA-compatible runtime (its last line of output is ``[ALINPU INFO] device
name=PPU-ZW810E``).  A native ``abort()`` does not unwind the Python stack, so
the benchmark produces **no JSON and no traceback**: the process is gone before
``run_benchmark`` can write anything.

What this restores
------------------
``faulthandler`` registers OS-level handlers for the fatal signals
(``SIGABRT``, ``SIGSEGV``, ``SIGFPE``, ``SIGBUS``, ``SIGILL``).  When the runtime
aborts, the handler dumps the Python stack of *every* thread -- including the
``model.generate`` worker thread the wrapper uses -- immediately before the
process dies.  That dump names the exact Python line the runtime aborted on,
which is the one thing a bare ``rc=134`` does not tell us.

The dump is written to a durable file next to the benchmark ``--output``
(``<output>.diag.log``) rather than to stderr, because the reported symptom is a
launcher that swallowed stderr -- the log stopped at ``[ALINPU INFO] device
name=PPU-ZW810E`` with nothing after it.  ``faulthandler`` supports a single
output sink, so the file wins over stderr here.  The stage markers written to
the same file also survive the abort, so the last marker identifies which
sample (warmup vs. scored, and its id) was in flight when the process died.

This module never suppresses or "handles" the abort -- it only makes it
legible.  It is safe to leave on: ``faulthandler`` costs nothing until a fatal
signal actually fires.
"""

from __future__ import annotations

import faulthandler
import os
import sys
import time
from pathlib import Path
from typing import Any, TextIO

#: Set to a falsy value to skip installing the fault handler and stage markers.
DIAGNOSTICS_ENV = "OPTIZ_QWEN_CRASH_DIAGNOSTICS"

#: Explicit override for the mirror-log path; otherwise derived from --output.
FAULT_LOG_ENV = "OPTIZ_QWEN_FAULT_LOG"

#: Kept module-global so the mirror file stays open for the whole process.  A
#: closed file would make ``faulthandler``'s handler a no-op at abort time.
_fault_log: TextIO | None = None


def diagnostics_enabled() -> bool:
    value = os.environ.get(DIAGNOSTICS_ENV, "").strip().lower()
    if value == "":
        return True
    return value in {"1", "true", "yes", "on"}


def resolve_fault_log_path(output_path: Any | None) -> Path | None:
    """Where the crash mirror log should be written, or ``None`` to skip it."""

    override = os.environ.get(FAULT_LOG_ENV, "").strip()
    if override:
        return Path(override)
    if output_path is None:
        return None
    output = Path(output_path)
    return output.with_name(output.name + ".diag.log")


def install_crash_diagnostics(output_path: Any | None = None) -> Path | None:
    """Enable ``faulthandler`` on stderr and (if resolvable) a mirror file.

    Returns the mirror-log path actually opened, or ``None`` when diagnostics
    are disabled or no path could be derived.  Idempotent: a second call keeps
    the first mirror file rather than leaking handles.
    """

    global _fault_log
    if not diagnostics_enabled():
        return None

    # faulthandler supports exactly one output file: a later enable() replaces
    # the earlier one, it does not add a second sink.  So arm stderr first as a
    # baseline, then -- if a mirror path resolves -- re-point the handler at the
    # file.  The file is the channel that matters here: this bug's symptom is a
    # launcher that swallowed stderr (the log stopped at "[ALINPU INFO]"), and
    # the mirror survives that.
    if not faulthandler.is_enabled():
        faulthandler.enable(file=sys.stderr, all_threads=True)

    if _fault_log is not None:
        return Path(_fault_log.name)

    log_path = resolve_fault_log_path(output_path)
    if log_path is None:
        # No file to write; the stderr handler above is the whole coverage.
        return None
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Line-buffered so markers hit disk before a subsequent abort.
        handle = log_path.open("w", encoding="utf-8", buffering=1)
    except OSError:
        # A missing or read-only log path must never break the benchmark; the
        # stderr handler above still covers the abort.
        return None

    # Re-point the single faulthandler sink at the durable file.  The abort dump
    # (every thread's Python stack) now lands here even when stderr is lost.
    faulthandler.enable(file=handle, all_threads=True)
    _fault_log = handle
    _write(f"[optiz-diag] crash diagnostics armed -> {log_path}")
    _write(f"[optiz-diag] python={sys.version.split()[0]} pid={os.getpid()}")
    _write(
        "[optiz-diag] CUDA_LAUNCH_BLOCKING="
        f"{os.environ.get('CUDA_LAUNCH_BLOCKING', '<unset>')}"
    )
    return log_path


def _write(message: str) -> None:
    if _fault_log is not None:
        try:
            _fault_log.write(message + "\n")
            _fault_log.flush()
        except (OSError, ValueError):  # pragma: no cover - closed/failed handle
            pass


def stage(message: str) -> None:
    """Record a progress marker to stdout and the mirror log, both flushed.

    The flush is the point: the marker must be durable before the next device
    call, so that when the runtime aborts the last surviving marker names the
    stage that triggered it.
    """

    if not diagnostics_enabled():
        return
    line = f"[optiz-diag] {time.strftime('%H:%M:%S')} {message}"
    try:
        print(line, flush=True)
    except (OSError, ValueError):  # pragma: no cover - closed stdout
        pass
    _write(line)


def log_runtime_environment(model: Any) -> None:
    """Record backend/device/dtype facts that shape which kernels dispatch.

    A vision-stage abort usually turns on the attention backend and dtype, so
    capturing them next to the crash marker removes a round-trip when reading
    the log afterwards.
    """

    if not diagnostics_enabled():
        return
    facts = [f"backend={getattr(model, 'backend_name', '?')}"]
    facts.append(f"dtype={getattr(model, 'dtype_name', '?')}")
    inner = getattr(model, "_model", None)
    config = getattr(inner, "config", None)
    if config is not None:
        facts.append(f"attn_impl={getattr(config, '_attn_implementation', '?')}")
        text_config = getattr(config, "text_config", None)
        if text_config is not None:
            facts.append(
                f"text_attn_impl={getattr(text_config, '_attn_implementation', '?')}"
            )
        vision_config = getattr(config, "vision_config", None)
        if vision_config is not None:
            facts.append(
                f"vision_attn_impl={getattr(vision_config, '_attn_implementation', '?')}"
            )
    for key in ("OPTIZ_QWEN_ATTN_PREFILL", "OPTIZ_QWEN_ATTN_DECODE"):
        value = os.environ.get(key)
        if value:
            facts.append(f"{key}={value}")
    stage("runtime " + " ".join(facts))
