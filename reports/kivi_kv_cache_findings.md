# Qwen3.5-2B KV 链路适配与对比报告

## 背景

当前主线是 **KV-only** 优化，不做视觉剪枝。目标是保留默认 baseline 不变，同时把 KV 相关链路拆成两条可选路径：

1. `legacy_kivi`：保留现有 KIVI 适配；
2. `qserve_kv`：实现一条独立的新 KV 链路，作为另一条实验分支。

## 当前实现状态

| 链路 | 状态 | 说明 |
| --- | --- | --- |
| Baseline | 保持不变 | 默认 `transformers` 路径 |
| legacy_kivi | 已保留 | 仍通过 `OPTIZ_QWEN_KIVI_KV_CACHE=1` 启用 |
| qserve_kv | 已落地 | 作为独立链路启用，采用本地 4-bit KV cache 实现 |

## 已验证结果

目前已完成的验证是结构与回归验证，不是最终性能胜出验证：

| 组别 | 验证项 | 结果 |
| --- | --- | --- |
| Baseline | 默认路径不受影响 | 通过 |
| legacy_kivi | 旧链路开关仍可路由 | 通过 |
| qserve_kv | 新链路可独立构建并进入调度 | 通过 |
| 测试 | `pytest tests/test_qserve_kv_cache.py tests/test_qwen35_kivi_cache.py tests/test_prefill_decode_scheduler.py` | `13 passed, 1 skipped` |

## 对比图

### 旧版 baseline vs KIVI 结果图

![Baseline vs KIVI](benchmarks/output/compare_baseline_vs_kivi_cn_10x64_validation_fix.png)

## 结果说明

1. 旧 KIVI 路径此前已经跑通，但没有赢过 baseline。

   旧结果中，Baseline 的 `TTFT` 和吞吐都优于 KIVI，说明“能跑”不等于“能赢”。

2. 当前新链路不是旧实现换名，而是独立 cache。

   `qserve_kv` 现在是单独的 `QServeKvCache`，并通过统一的 KV chain 分发进入 wrapper。

3. 目前还不能宣称性能收益。

   现阶段只验证了链路分离、回归稳定和测试通过，尚未完成服务器上的真实 benchmark 对比。

## 结论

- 已完成：旧 KIVI 保留为可选链路，新 `qserve_kv` 作为另一条链路落地；
- 已验证：入口、调度、测试回归通过；
- 未完成：服务器真实 benchmark 的性能胜出验证；
- 当前策略：继续把 `qserve_kv` 作为主优化试验链路，baseline 和 legacy_kivi 保持可对照。
