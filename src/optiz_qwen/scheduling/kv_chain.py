"""KV-only chain selection for Qwen3.5-2B optimization experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from optiz_qwen.compression import (
    KiviConfig,
    QServeKvConfig,
    build_qwen35_kivi_cache,
    build_qserve_kv_cache,
)
from optiz_qwen.scheduling.prefill_decode import PrefillDecodeStats


@dataclass(frozen=True)
class KVChainReport:
    """Metadata describing which KV optimization chain ran."""

    chain_name: str
    enabled: bool
    k_bits: int | None = None
    v_bits: int | None = None
    group_size: int | None = None
    residual_length: int | None = None
    implementation: str | None = None


def build_kv_chain(
    *,
    chain_name: str,
    model_config: Any,
    enabled: bool,
    kivi_config: KiviConfig | None = None,
    qserve_config: QServeKvConfig | None = None,
) -> tuple[Any | None, KVChainReport]:
    """Build the selected KV chain object and a compact report."""

    normalized = chain_name.strip().lower()
    if not enabled:
        return None, KVChainReport(chain_name=normalized, enabled=False)

    if normalized == "legacy_kivi":
        cfg = kivi_config or KiviConfig()
        cache = build_qwen35_kivi_cache(model_config, cfg)
        return cache, KVChainReport(
            chain_name=normalized,
            enabled=True,
            k_bits=cfg.k_bits,
            v_bits=cfg.v_bits,
            group_size=cfg.group_size,
            residual_length=cfg.residual_length,
            implementation="legacy_kivi",
        )

    if normalized == "qserve_kv":
        cfg = qserve_config or QServeKvConfig()
        cache = build_qserve_kv_cache(model_config, cfg)
        report = KVChainReport(
            chain_name=normalized,
            enabled=True,
            k_bits=cfg.k_bits,
            v_bits=cfg.v_bits,
            group_size=cfg.group_size,
            residual_length=cfg.residual_length,
            implementation="qserve_kv4_local",
        )
        return cache, report

    raise ValueError(f"unsupported KV chain: {chain_name}")
