"""PPU SDK environment bootstrap.

Why this module exists
----------------------
The PPU (PPU-ZW810E / ALINPU) runtime compiles device kernels at first use
(RTC -- runtime compilation).  RTC locates its toolchain through the ``PPU_SDK``
/ ``PPU_HOME`` environment variables that ``/usr/local/PPU_SDK/envsetup.sh``
exports.  When a process is launched from a bare shell that never sourced
``envsetup.sh``, the very first kernel compilation aborts::

    [ERROR ... rtc_kernel.cu:440] Both PPU_SDK and PPU_HOME are not exist

The native ``abort()`` (SIGABRT, rc=134) unwinds nothing, so the benchmark dies
in the first vision forward with no JSON and no traceback -- the exact reported
symptom.  This was diagnosed with the crash-diagnostics faulthandler: the dumped
stack landed on a plain ``nn.Linear.forward`` (the vision patch-merger), which
is not an unsupported op -- it was the first kernel RTC tried to build.

What this restores
------------------
``ensure_ppu_sdk_env`` sets ``PPU_SDK``/``PPU_HOME`` (and the CUDA-shim paths
``envsetup.sh`` exports in CUDA mode) *before any device work*, but only when
they are unset and a real SDK directory is present.  Setting ``PPU_SDK`` from
within the process before the first kernel is sufficient -- verified on target:
RTC reads the variable at compile time, so an in-process assignment is honoured.

It is a strict no-op when:

- ``PPU_SDK`` or ``PPU_HOME`` is already set (a sourced ``envsetup.sh`` wins),
- no SDK directory can be found (non-PPU hosts, local Windows dev),
- disabled explicitly via ``OPTIZ_QWEN_PPU_SDK_BOOTSTRAP=0``.

It never overrides an operator's existing environment and never raises.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: Set to a falsy value to skip the bootstrap entirely.
BOOTSTRAP_ENV = "OPTIZ_QWEN_PPU_SDK_BOOTSTRAP"

#: Explicit override for the SDK root; otherwise the well-known paths are tried.
SDK_ROOT_ENV = "OPTIZ_QWEN_PPU_SDK_ROOT"

#: Well-known SDK install locations, in priority order.  ``envsetup.sh`` lives
#: at ``<root>/envsetup.sh``; its presence is how a candidate is validated.
_CANDIDATE_ROOTS = (
    "/usr/local/PPU_SDK",
)


@dataclass
class PpuSdkBootstrap:
    """Outcome of :func:`ensure_ppu_sdk_env`, for logging and tests."""

    applied: bool
    reason: str
    sdk_root: str | None = None
    variables_set: dict[str, str] = field(default_factory=dict)


def _existing_sdk_root() -> Path | None:
    override = os.environ.get(SDK_ROOT_ENV, "").strip()
    candidates = [override] if override else list(_CANDIDATE_ROOTS)
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate)
        # envsetup.sh is the SDK's own marker file; requiring it avoids
        # binding to an empty or half-populated directory.
        if (root / "envsetup.sh").is_file():
            return root
    return None


def bootstrap_enabled() -> bool:
    value = os.environ.get(BOOTSTRAP_ENV, "").strip().lower()
    if value == "":
        return True
    return value in {"1", "true", "yes", "on"}


def ensure_ppu_sdk_env() -> PpuSdkBootstrap:
    """Set ``PPU_SDK``/``PPU_HOME`` before first kernel use, if needed.

    Idempotent and side-effect-free when a valid SDK env is already present or
    no SDK is installed.  Returns a :class:`PpuSdkBootstrap` describing what was
    done so callers can record it next to the crash-diagnostics markers.
    """

    if not bootstrap_enabled():
        return PpuSdkBootstrap(applied=False, reason="disabled via env")

    # A sourced envsetup.sh (either mode) is authoritative -- do not touch it.
    if os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME"):
        return PpuSdkBootstrap(
            applied=False,
            reason="PPU_SDK/PPU_HOME already set",
            sdk_root=os.environ.get("PPU_SDK") or os.environ.get("PPU_HOME"),
        )

    root = _existing_sdk_root()
    if root is None:
        return PpuSdkBootstrap(applied=False, reason="no SDK directory found")

    root_str = str(root)
    variables: dict[str, str] = {}

    # PPU_SDK is the one RTC actually checks; PPU_HOME mirrors it because the
    # abort message names both and either satisfies the check.
    variables["PPU_SDK"] = root_str
    variables["PPU_HOME"] = root_str

    # Mirror the CUDA-shim paths envsetup.sh exports in CUDA mode, but only when
    # the shim directory exists.  These are not required to clear the abort
    # (PPU_SDK alone suffices on target) yet keep downstream toolchain lookups
    # consistent with a properly sourced envsetup.sh.
    cuda_sdk = root / "CUDA_SDK"
    if cuda_sdk.is_dir():
        for key in ("CUDA_PATH", "CUDA_HOME", "CUDA_TOOLKIT_ROOT"):
            if not os.environ.get(key):
                variables[key] = str(cuda_sdk)

    for key, value in variables.items():
        os.environ[key] = value

    return PpuSdkBootstrap(
        applied=True,
        reason="set from installed SDK",
        sdk_root=root_str,
        variables_set=variables,
    )
