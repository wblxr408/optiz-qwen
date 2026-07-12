# QServe fused KV 技术决策与实测记录

## 决策范围

目标是在保持 baseline 默认行为不变的前提下，同时改善 Qwen3.5-2B 的 TTFT 与单样本 decode 吞吐。

## 方案对比

| 方案 | 性能潜力 | 实现成本 | 兼容性 | 选择结论 |
| --- | --- | --- | --- | --- |
| 继续调整 `qserve_kv4_local` 参数 | 低；仍逐 token 全量反量化 | 低 | 高 | 否决，无法消除根本开销 |
| 仅注册 Transformers Attention backend | 中 | 中 | 中 | 否决，cache `update()` 在 backend 前已要求完整 K/V |
| 修改整个 Qwen3.5 模型类 | 高 | 高 | 低 | 否决，容易随 Transformers 主线漂移 |
| 模型实例级 full-attention 适配 + Triton packed-KV kernel | 高 | 高 | 中 | 采用，baseline 和 linear-attention 层保持不变 |
| 固定 token ID 删除视觉 token | 中 | 中 | 低 | 否决，会破坏 M-RoPE 和视觉网格对齐 |
| 官方 image processor 视觉像素预算 | 中 | 低 | 高 | 采用为可选 TTFT 路径，必须做 OCR/定位精度扫描 |

最终实现保留 `qserve_kv`，新增 `qserve_fused_kv`；两者均为 opt-in。融合链只有在结果元数据出现
`active_backend=triton_int4_decode` 且 `kernel_calls>0` 时，才证明真实走过 packed-KV kernel。

## 当前实测

数据均来自同一台 RTX 4070 Laptop GPU 的 1-sample smoke，仅用于链路诊断，不能代表 10-sample 或 PPU 结论。

| 配置 | Runner | TTFT (ms) | 吞吐 (tokens/s) | 准确率 | 说明 |
| --- | --- | ---: | ---: | ---: | --- |
| baseline | `greedy` | 4681.553 | 5.797 | 1.0 | 公平 runner 对照 |
| `qserve_fused_kv` | `greedy` | 4154.487 | 5.208 | 1.0 | TTFT 改善 11.26%，吞吐下降 10.16% |
| `qserve_fused_kv` | `generate` | 5266.158 | 4.726 | 1.0 | 不作为推荐配置 |
| 视觉预算 36864 | `generate` | 4684.236 | 5.187 | 1.0 | 单样本保持正确，收益尚不稳定 |

融合 kernel 的独立数值验证相对同一 INT4 缓存的 eager 参考输出：最大绝对误差 `2.44e-4`，平均绝对误差
`2.77e-5`。真实 fused smoke 记录 `378` 次 kernel 调用、`0` 次 fallback。

## 10-sample 主线结果

在相同公开中文数据、相同 10 个样本、`2` 个 warmup、`max_new_tokens=64` 和原生 `generate` runner 下：

| 配置 | TTFT (ms) | 吞吐 (tokens/s) | 准确率 |
| --- | ---: | ---: | ---: |
| baseline | 1612.735 | 5.562 | 1.0 |
| 视觉预算 36864 (`192x192`) | 888.455 | 10.346 | 1.0 |
| 提升 | 44.91% | 86.01% | 持平 |

逐样本检查显示 10/10 TTFT 均改善、10/10 吞吐均提升，输出 token 数没有统一提前结束；TTFT 中位数
`1514.022 -> 841.111 ms`，吞吐中位数 `5.267 -> 9.814 tokens/s`。对应 artifact：

- `benchmarks/output/baseline_generate_cuda_cn_10x64_20260712.json`
- `benchmarks/output/visual_budget_192_generate_cuda_cn_10x64_20260712.json`

因此当前推荐主线是视觉预算 36864，不包含短上下文下仍拖慢吞吐的 `qserve_fused_kv`。融合 KV 保留为长上下文和 PPU
带宽实验链，不替换已经实测胜出的主线。

## 结论和下一门槛

当前已在 10-sample public smoke 上得到 TTFT 与吞吐同时胜出的配置。`qserve_fused_kv` 是已实现、已数值验证、已真实运行的
实验链，但在当前 304-token 短上下文上存在性能不达标问题，不能设为默认或宣称优化胜出。

剩余门槛是扩大 public-dev 数据集，单独统计 OCR 和细粒度定位类别，并在 PPU 上复测。PPU 侧尚未验证，因此没有
“PPU-adapted”结论。
