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

### 6.3 零额外权重副本正式实现

正式实现不再长期保留原四份权重和一份拼接副本。安装时先构造 packed 权重，再把原有
QKV、Z、B、A 四个 `nn.Linear.weight` 重新绑定为该存储的切片视图，因此
`state_dict` 的键名和张量形状保持不变，多 token prefill 仍执行原四投影路径。

18 层共 72 个投影的原权重与 packed 权重均为 606,339,072 bytes（578.25 MiB）。PPU
实测安装前后 `torch.cuda.memory_allocated` 均为 8,853,537,792 bytes，差值严格为 0。

去除热路径中的重复形状检查和统计锁后，中英文各 150 题最终配对结果如下：

| 数据集 | 指标 | Baseline | 零副本融合 | 变化 |
|---|---|---:|---:|---:|
| EN150 | Accuracy | 75.333% | 75.333% | 0 |
| EN150 | TTFT | 150.413 ms | 148.833 ms | -1.05% |
| EN150 | Throughput | 71.844 tok/s | 73.284 tok/s | +2.00% |
| EN150 | Elapsed | 341.977 ms | 337.809 ms | -1.22% |
| CN150 | Accuracy | 79.333% | 79.333% | 0 |
| CN150 | TTFT | 119.539 ms | 119.379 ms | -0.13% |
| CN150 | Throughput | 53.078 tok/s | 54.035 tok/s | +1.80% |
| CN150 | Elapsed | 926.864 ms | 909.312 ms | -1.89% |

300 题的选项均无分歧；解释文本仍有 EN 1 题、CN 10 题措辞不同，与临时原型一致。
该结果证明零副本版本保留了约 2% 的 decode 收益，但也进一步确认投影融合不是 TTFT
大幅优化的来源。

### 6.4 PPU prefill Delta Rule 内核

正式目标是优化 Transformers 调用的 chunk Delta Rule 路径。当前第一版设备实现并非论文式
分块算法，而是针对 Qwen3.5-2B 固定 `H=16, K=128, V=128` 的直接递推核：每个 block
负责一个 head/value 通道，128 个线程并行维护 key 维状态。该结构在 PPU 上已经显著快于
当前 PyTorch chunk fallback，因此值得作为第一版后端保留。

FP32 点积归约虽然达到 4.46--6.33 倍微基准加速，但输出误差约 0.0026--0.0039、状态误差
约 0.0085--0.0180，并在 EN20 造成 1 题由对变错，因此拒绝。改为 FP64 点积归约、保持
FP32 recurrent state 后：

| 序列长度 | 官方 fallback | PPU kernel | 加速 | 输出最大误差 | 状态最大误差 |
|---:|---:|---:|---:|---:|---:|
| 337 | 3.819 ms | 1.448 ms | 2.64x | 0.000488 | 1.37e-6 |
| 360 | 3.829 ms | 1.551 ms | 2.47x | 0.000488 | 1.37e-6 |
| 512 | 4.132 ms | 2.192 ms | 1.88x | 0.000488 | 2.03e-6 |

全 18 层替换仍会累积数值扰动：EN150 有 3 个选项分歧、准确率净增 1 题；CN150 有
5 个选项分歧、准确率净减 1 题。采用与题目内容无关的“仅最后 9 个 GDN 层”策略后：

| 数据集 | 指标 | Baseline | 后 9 层 PPU kernel | 变化 |
|---|---|---:|---:|---:|
| EN150 | Accuracy | 75.333% | 76.000% | +0.67 pp |
| EN150 | TTFT | 153.320 ms | 123.985 ms | -19.13% |
| EN150 | Throughput | 70.166 tok/s | 70.025 tok/s | -0.20% |
| EN150 | Elapsed | 348.587 ms | 319.749 ms | -8.27% |
| CN150 | Accuracy | 79.333% | 80.000% | +0.67 pp |
| CN150 | TTFT | 122.286 ms | 93.442 ms | -23.59% |
| CN150 | Throughput | 52.156 tok/s | 52.085 tok/s | -0.14% |
| CN150 | Elapsed | 945.542 ms | 935.697 ms | -1.04% |

两种语言各有 1 个选项分歧，均为 baseline 错误或无法解析、kernel 版本回答正确；本轮
300 题未观察到准确率回归。不过 EN 有 2 个、CN 有 40 个解释文本分歧，因此该实现不能
宣称逐位等价，必须保持默认关闭并通过显式开关启用。

## 8. 工程限制

6.2 的临时原型通过拼接 18 层权重构建融合矩阵，曾额外占用约 **578.25 MiB** BF16
显存。6.3 已采用“一份 packed 权重作为唯一存储、原投影使用切片视图”的方案消除该
副本。正式实现还有两项约束：

1. 必须在模型到达最终 device 和 dtype 后安装；安装后再次迁移会立即报错；
2. 当前收益来自一次大 `F.linear`，后续仍可比较 PPU 融合 GEMV 或完整 decode
   CUDAGraph 的边际收益。

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
  TTFT 潜力的目标。首个 PPU 直接递推核已将后 9 层方案的 TTFT 降低 19%--24%，证明
  路线有效；下一步应比较真正的 chunk/blocked PPU 算法能否在不扩大误差的前提下继续
  提升长序列性能。
- **已完成内核契约与正式后端**：仓库包含与 Transformers fallback 对齐的 FP32 状态
  reference、误差比较器、HGGC/CUDA-compatible 设备源码、显式层选择和默认关闭开关。

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

零额外权重副本正式实现的最终证据位于
`benchmarks/output/ppu_gdn_projection_fusion_20260801/`：

- `en150.json`
- `cn150.json`

PPU Delta Rule 的微基准、全 18 层结果和后 9 层结果位于
`benchmarks/output/ppu_chunk_delta_20260801/`，主要文件包括：

- `microbenchmark_fp32_reduction.json`
- `microbenchmark_fp64_reduction.json`
- `en150_fp64_reduction.json`
- `cn150_fp64_reduction.json`
- `en150_last9.json`
- `cn150_last9.json`

该目录属于 benchmark 输出，不随 Git 跟踪；报告中的数值均来自上述实际运行结果。
