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
| + PPU delta kernel（bf16，后 9 层） | 142.4* | 60.8 | 74% |
| + KV 链 INT4（bf16，greedy） | 129.5 | 15.1 | 74% |
| + ToME L16R32（bf16，推荐配置） | 249.2 | 61.3 | 78% |
| + GDN decode fusion（bf16） | 176.2 | 60.5 | 74% |

CN50 基线（fp16）：TTFT 137.9 ms / 44.3 tok/s / 86%。

## 关键发现

1. **PPU delta kernel 在真机上有效**：预热后 TTFT 从 173.9 ms 降至 142.4 ms
   （-18%），与 8/1 报告 EN150 -19.1% 一致。首次运行存在冷启动异常样本
   （2.7 s 峰值，内核编译），第二次运行同批样本全部恢复正常（max 165 ms）。
   表中 142.4 ms 为预热后稳态值（`delta_bf16_en50_run2_warm.json`）。
2. **KV 链 TTFT 收益显著但解码吞吐代价大**：TTFT -26%（174→130 ms），但 INT4
   Triton 分割解码把吞吐从 62 打到 15 tok/s（-76%）。是否采用取决于评分权重。
3. **ToME L16R32（报告推荐配置）在 PPU 上无 TTFT 收益**：TTFT +42%，但准确率
   78%（+2pt，本轮最高）。作为速度优化不采用；准确率提升可能来自样本波动，
   需更大样本确认，另行记录。
4. **GDN decode fusion 无端到端收益**：确认生效（1584 次融合调用）但性能持平。
5. **dtype**：bf16 与 fp16 性能几乎相同；fp16 准确率略高（76% vs 74%），
   官方默认 fp16 合理。delta 内核要求 bf16，因此启用时需配合
   `OPTIZ_QWEN_TORCH_DTYPE=bf16`。

## 环境备注

- PPU delta 内核需要 bf16 输入（CUDA 扩展硬性检查），wrapper 已新增
  `OPTIZ_QWEN_TORCH_DTYPE` 环境变量开关，默认仍为 fp16，不改变既有行为。
- 实例本地依赖：`transformers==5.14.1` + 阿里云 pip 源；PPU SDK 需每次
  `source /usr/local/PPU_SDK/envsetup.sh`。

## 勘误

初版测试曾报告 delta kernel "TTFT +42%"，原因是首次运行的 2 条冷启动样本
（q=269、q=344，约 2.7 s）污染了平均值；第二次运行同配置后平均值恢复至
142.4 ms，两条样本分别为 138.8 ms 与 142.1 ms。冷启动问题与 8/1 报告
"PPU 对动态形状存在首次编译开销"一致。

## Delta 微基准与环境一致性

与 8/1 报告第 6.4 节同口径（BF16、H=16/K=128/V=128、FP64 归约）重测：

| seq | 8/1 kernel | 8/16 kernel | 8/1 加速比 | 8/16 加速比 |
|---|---:|---:|---:|---:|
| 337 | 1.448 ms | 1.533 ms | 2.64× | 3.69× |
| 360 | 1.551 ms | 1.619 ms | 2.47× | 3.48× |
| 512 | 2.192 ms | 2.274 ms | 1.88× | 2.64× |

PPU delta kernel 本身耗时与 8/1 逐点一致（误差 <6%），验证新实例环境与
公共服务器一致。加速比口径差异源于 baseline 取完整 GDN 层前向（含投影）
而非 delta rule 子模块。
