# PPU Gated DeltaNet 瓶颈与融合路线验证

日期：2026-08-01

设备：PPU-ZW810E

模型：Qwen3.5-2B，BF16，Transformers 5.14.1，PyTorch 2.9.0+PPU

## 1. 实验目的

文献综述将 Gated DeltaNet（GDN）快速路径、算子融合和计算图捕获列为最高优先级。
本轮使用学校 PPU 回答四个问题：

1. 视觉编码器、GDN、全注意力和 MLP 中谁是实际瓶颈；
2. GDN 内部具体慢在哪一段；
3. PPU 的计算图捕获是否真正可用；
4. 一个简单、低风险的 GDN 投影融合能否改善端到端指标并保持答案。

服务器上的官方环境仍明确提示缺少 FLA 和 `causal_conv1d` 快速路径，Qwen3.5-2B
实际走 Transformers 的普通 PyTorch fallback。本轮只上传临时通用探针，实验结束后已清理。

## 2. 模块级瓶颈

使用 3 条英文公开样本，分别记录预填充和单 token 解码调用。模型结构检查确认包含
18 个 `Qwen3_5GatedDeltaNet`、6 个 `Qwen3_5Attention`、24 个 MLP 和 1 个视觉编码器。

| 模块 | 阶段 | 调用数 | 总计时 | 单次中位数 |
|---|---|---:|---:|---:|
| Gated DeltaNet | prefill | 54 | 253.393 ms | 4.589 ms |
| Full Attention | prefill | 18 | 11.390 ms | 0.588 ms |
| MLP | prefill | 72 | 27.178 ms | 0.347 ms |
| Vision Encoder | vision | 3 | 36.715 ms | 12.149 ms/样本 |
| Gated DeltaNet | decode | 846 | 510.352 ms | 0.593 ms |
| Full Attention | decode | 282 | 129.896 ms | 0.456 ms |
| MLP | decode | 1128 | 89.201 ms | 0.077 ms |

同步计时会扰动绝对端到端时间，因此这些值不能直接相加当作 benchmark 总耗时；但它们
足以判断相对瓶颈。GDN 占已计时语言模块 prefill 的约 86.8%，占 decode 的约 70.0%。
视觉编码器平均仅约 12.2 ms，再次解释了视觉 token 路线为何难以显著改善总指标。

**结论：GDN 是 PPU 上的第一瓶颈，优化方向由文献判断升级为目标硬件实证。**

## 3. GDN 内部拆分

使用 2 条英文样本进一步拆分 18 层 GDN。prefill 长度分别为 337 和 360，decode
长度为 1。

| 子模块 | prefill 单层平均 | decode 单层平均 | 主要判断 |
|---|---:|---:|---|
| Delta Rule | 4.212 ms | 0.255 ms | prefill 的绝对主瓶颈 |
| Causal Conv | 0.087 ms | 0.086 ms | decode 中值得融合 |
| Gated RMSNorm | 0.123 ms | 0.099 ms | decode 中的小算子热点 |
| QKV/Z/A/B 四投影 | 0.254 ms | 0.095 ms | 可减少 launch 次数 |
| Output Projection | 0.090 ms | 0.037 ms | 次要 |

在已拆出的 GDN 子模块中，Delta Rule 占 prefill 约 88.3%；decode 更分散，Delta Rule、
门控归一化、卷积更新和四投影均有可观占比。

**结论：prefill 应优先实现 PPU 版 chunk Delta Rule；decode 应采用计算图和多算子融合。**

## 4. PPU 计算图可行性

对 BF16、静态 `[1, 1, 2048]` 输入的两层通用 MLP 进行 1000 次调用：

| 路径 | 单次时间 | 数值误差 |
|---|---:|---:|
| Eager | 29.017 us | - |
| CUDAGraph replay | 15.171 us | 0 |

微基准加速为 **1.913 倍**。这不能直接代表完整模型，因为真实 GDN cache 和生成长度
具有动态状态；但它证明 PPU 的 `CUDAGraph` 不只是暴露了接口，而是能够正确捕获、重放
并减少静态小算子链的调度开销。

## 5. 四投影融合微基准

GDN 对同一 `hidden_states` 连续执行 QKV、Z、B、A 四次无 bias 线性投影。实验将四份
权重按输出维拼接，改为一次 `F.linear` 后按 `[6144, 2048, 16, 16]` 拆分。

| 序列长度 | Separate eager | Fused eager | Eager 加速 | 图内融合加速 | 数值结果 |
|---:|---:|---:|---:|---:|---|
| 1 | 0.033217 ms | 0.013368 ms | 2.485× | 1.645× | 完全一致 |
| 360 | 0.147810 ms | 0.105513 ms | 1.401× | 1.374× | 最大误差 0.0078125 |

单 token 解码的融合逐位一致；长 prefill 会因 GEMM 形状变化产生很小的 BF16 舍入差异。

## 6. 端到端候选比较

### 6.1 全阶段融合：拒绝

先在 prefill 和 decode 均启用融合。英文 50 题中：

| 指标 | Baseline | 全阶段融合 | 变化 |
|---|---:|---:|---:|
| Accuracy | 74% | 72% | -2 个百分点 |
| TTFT | 152.647 ms | 151.748 ms | -0.59% |
| Throughput | 69.928 tok/s | 70.470 tok/s | +0.78% |
| 答案分歧 | - | 1/50 | 不可接受 |

样本 404 从正确答案 B 变为 A。局部 BF16 误差虽然极小，仍可能沿 24 层和自回归生成放大。
该版本收益很小且影响答案，不应进入主线。

### 6.2 仅 decode 融合：保留候选

保守版本只在 `seq_len == 1` 时融合，prefill 完全使用原路径。中英文各 150 题的配对结果：

| 数据集 | 指标 | Baseline | Decode-only | 变化 |
|---|---|---:|---:|---:|
| EN150 | Accuracy | 75.333% | 75.333% | 0 |
| EN150 | TTFT | 152.069 ms | 151.008 ms | -0.70% |
| EN150 | Throughput | 70.820 tok/s | 71.726 tok/s | +1.28% |
| EN150 | Elapsed | 346.758 ms | 341.885 ms | -1.41% |
| CN150 | Accuracy | 79.333% | 79.333% | 0 |
| CN150 | TTFT | 124.526 ms | 121.246 ms | -2.63% |
| CN150 | Throughput | 51.682 tok/s | 53.237 tok/s | +3.01% |
| CN150 | Elapsed | 951.889 ms | 921.527 ms | -3.19% |

300 题的最终选项保持一致。英文有 1 题、中文有 10 题的解释文本措辞不同，因此目前只能
表述为“300 题选项和准确率保持”，不能声称全模型逐位无损。中文集基线与优化版均有
14 条 `missing_choice_answer`，属于当前官方解析口径下的共同问题，不是融合新增回归。

中文平均生成约 42.6 token，英文约 12.7 token。decode 更长时收益更明显，符合该优化只
作用于单 token 解码的机制。配对 bootstrap 结果也显示：中文 TTFT、吞吐和总时长的
95% 区间均为正向；英文 TTFT 和吞吐区间仍跨过 0，英文收益证据较弱。

## 7. 控制组与噪声

为避免将两份模型实例的自然波动误判为优化收益，额外运行 baseline 对 baseline 控制组：

| 控制组 | TTFT 差异 | Throughput 差异 | Elapsed 差异 | 文本/答案分歧 |
|---|---:|---:|---:|---:|
| EN50 | +0.32% | -0.34% | +0.72% | 0 / 0 |
| CN50 | +0.24% | -0.19% | -0.18% | 0 / 0 |

仅 decode 融合的中文 +3.01% 吞吐明显高于控制组波动；英文 +1.28% 也高于本轮控制组，
但优势较小，需要更多重复实验才能给出更强的统计结论。

## 8. 工程限制

当前临时原型通过拼接 18 层权重构建融合矩阵，额外占用约 **578.25 MiB** BF16 显存。
这适合验证，不适合最终提交。正式实现应采用以下之一：

1. 用一份 packed 权重作为唯一存储，prefill 的四次投影使用其切片视图；
2. 编写 PPU 融合 GEMV 内核，直接读取原四份权重并一次调度输出；
3. 将投影融合纳入完整 decode CUDAGraph，比较融合在图捕获后的边际收益。

此外，当前 wrapper 使用 Python streamer，TTFT 会包含线程与文本块输出行为。后续需要
在模型 forward 边界增加设备事件计时，区分真实首 token 完成与 streamer 可见时间。

## 9. 决策与下一步

### 保留

- **仅 decode 四投影融合**：已有 300 题选项保持和跨语言正向性能证据；应重写为零额外
  权重副本、可开关的正式实现。
- **GDN decode CUDAGraph**：微基准可用，且小算子链占比高；应在固定 batch=1、seq=1
  和静态 cache 地址下做整层捕获。

### 最高收益主线

- **PPU chunk Delta Rule 内核**：占已拆出 GDN prefill 的约 88.3%，是唯一具有大幅降低
  TTFT 潜力的目标。应参考 Gated DeltaNet、FLA 和 FlashQLA 的分块算法，但使用 PPU SDK
  实现，不照搬 CUDA/Triton 代码。

### 拒绝

- **prefill 四投影融合**：局部更快但已造成 1/50 答案回归；除非解决数值一致性，不再投入。
- **继续微调视觉 token 剪枝**：PPU 画像再次证明视觉编码不是主要端到端瓶颈。

## 10. 原始证据

本地原始 JSON 位于 `benchmarks/output/ppu_module_profile_20260801/`，包括：

- `module_profile_en3.json`
- `gdn_breakdown_en2.json`
- `cudagraph_mlp_probe.json`
- `projection_fusion_microbenchmark.json`
- `fused_projection_en50.json`
- `decode_fused_projection_en50.json`
- `decode_fused_projection_en150.json`
- `decode_fused_projection_cn150.json`
- `control_baseline_en50.json`
- `control_baseline_cn50.json`

该目录属于 benchmark 输出，不随 Git 跟踪；报告中的数值均来自上述实际运行结果。
