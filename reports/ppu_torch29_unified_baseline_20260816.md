# PPU torch2.9 统一口径基线与优化开关实测

日期：2026-08-16

环境：新实例 `ppu-training` 镜像（PyTorch 2.9.0 + CUDA 13.0 + Transformers 5.14.1，
PPU-SDK 2.1.0），真机 PPU-ZW810E。

目的：统一使用 `dndx_public_benchmark.py` 作为评测口径，重新测量基线和仓库全部
优化开关在 PPU 真机上的端到端表现，替代旧服务器上不同评测器产生的历史数字。

## 评测口径

- 评测器：`optiz_qwen.evaluation.dndx_public_benchmark`（官方 v1.2 协议）
- 数据：MMBench dev EN/CN 各 50 条，warmup 2 条，`max_new_tokens=256`，seed 默认
- 结果 JSON 保留在本地 `benchmarks/output/ppu_torch29_baseline_20260816/`
  （按仓库惯例被 Git 忽略）

## EN50 结果

| 配置 | TTFT (ms) | 吞吐 (tok/s) | 准确率 |
|---|---:|---:|---:|
| 基线 fp16（官方默认） | 175.8 | 60.3 | 76% |
| 基线 bf16 | 173.9 | 62.0 | 74% |
| + PPU delta kernel（bf16，后 9 层） | 246.2 | 61.1 | 74% |
| + KV 链 INT4（bf16，greedy） | 129.5 | 15.1 | 74% |
| + ToME L8/R32（bf16） | 232.0 | 61.6 | 66% |
| + GDN decode fusion（bf16） | 176.2 | 60.5 | 74% |

CN50 基线（fp16）：TTFT 137.9 ms / 44.3 tok/s / 86%。

## 关键发现

1. **PPU delta kernel 在真机上拖慢 prefill**：TTFT 从 174 ms 增至 246 ms（+42%），
   但确认内核被调用（27 次 prefill 调用）。该内核需要针对 PPU 重新审视实现
   （当前为 fp64 归约路径），在优化完成前不建议启用。
2. **KV 链 TTFT 收益显著但解码吞吐代价大**：TTFT -26%（174→130 ms），但 INT4
   Triton 分割解码把吞吐从 62 打到 15 tok/s（-76%）。是否采用取决于评分权重。
3. **ToME L8/R32 与 8/1 结论一致**：准确率 -8pt 且 TTFT 变慢，不采用。
4. **GDN decode fusion 无端到端收益**：确认生效（1584 次融合调用）但性能持平。
5. **dtype**：bf16 与 fp16 性能几乎相同；fp16 准确率略高（76% vs 74%），
   官方默认 fp16 合理。delta 内核要求 bf16，因此启用时需配合
   `OPTIZ_QWEN_TORCH_DTYPE=bf16`。

## 环境备注

- PPU delta 内核需要 bf16 输入（CUDA 扩展硬性检查），wrapper 已新增
  `OPTIZ_QWEN_TORCH_DTYPE` 环境变量开关，默认仍为 fp16，不改变既有行为。
- 实例本地依赖：`transformers==5.14.1` + 阿里云 pip 源；PPU SDK 需每次
  `source /usr/local/PPU_SDK/envsetup.sh`。
