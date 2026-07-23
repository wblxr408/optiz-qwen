# 视觉 Token 剪枝适用性：论文 Token 规模与 Qwen3.5-2B 实测对比

## 结论摘要

现有主流视觉 Token 剪枝论文的显著收益，大多建立在视觉 Token 占输入序列绝对多数的模型上。典型设置为单图 `576` 或 `2880` 个视觉 Token，视频设置为 `1152` 或 `2048` 个视觉 Token。

我们的 Qwen3.5-2B 在 MMBench EN 50 条基线中，单图视觉 Token 数为 `64-192`，平均 `95.28`。它只有 LLaVA-1.5 常见设置的 `16.54%`，只有 LLaVA-NeXT 高分辨率设置的 `3.31%`。这说明 Qwen3.5-2B 在进入语言模型前已经进行了很强的视觉压缩，论文中最核心的前提——“视觉 Token 数量庞大并占据输入序列主体”——在我们的任务中明显减弱。

因此，当前证据不支持继续把中间层视觉 Token 剪枝作为主要性能突破口。它仍可作为研究性实验保留，但预期收益上限较低，而且排序、动态裁剪和 KV cache 维护的固定成本更容易覆盖节省的计算量。

## 我们的实测基线

数据来自：

- 结果文件：`benchmarks/output/result_dev_en_50_baseline_accuracy_mps.json`
- 样本数：`50`
- 设备：Apple MPS
- 最少视觉 Token：`64`
- 最多视觉 Token：`192`
- 平均视觉 Token：`95.28`
- 中位视觉 Token：`80`

这不是根据图像分辨率推算的理论值，而是 processor 实际生成并记录在每条 benchmark 结果中的 `image_token_count`。

## 论文模型与视觉 Token 规模

| 论文 / 方法 | 主要模型与输入 | 原始视觉 Token | 典型保留数 | 论文报告的收益 | 收益证据口径 |
|---|---|---:|---:|---|---|
| FastV | LLaVA-1.5，336×336 单图 | 576 | 288（保留 50%） | LLaVA-1.5-13B 图像相关 FLOPs 最高约下降 45% | 主要是理论 FLOPs；另有特定任务 latency |
| FastV | Video-LLaVA，视频 | 2048 | 1024（保留 50%） | FLOPs 降至约 52.3% | 理论 FLOPs |
| PyramidDrop | LLaVA-1.5 单图 | 576 | 分阶段约 576→288→144→72 | 推理 FLOPs 3.82T→1.78T | 理论 FLOPs；未充分报告端到端推理延迟 |
| PyramidDrop | LLaVA-NeXT 高分辨率单图 | 最高 2880 | 分阶段指数下降 | 推理 FLOPs 20.8T→9.46T | 理论 FLOPs；真实时间主要报告训练 GPU hours |
| SparseVLM | LLaVA-1.5 单图 | 576 | 192 / 128 / 64 等 | 约 37% CUDA latency 降低并保留约 97% 相对准确率 | A100 CUDA latency + FLOPs |
| VisionZip | LLaVA-1.5 单图 | 576 | 192 / 128 / 64 | 保留 64 时约保留 94% 平均性能 | 真实 GPU 推理与 prefill；同时报告 FLOPs/性能 |
| VisionZip | LLaVA-NeXT 高分辨率单图 | 2880 | 640 / 320 / 160 | prefill 最多约 8×，GPU inference 约 2× | 真实 GPU 时间；高 Token 场景 |
| FasterVLM | LLaVA-1.5 单图 | 576 | 288 / 144 / 58 / 29 | 高剪枝率下优于 FastV；50% 剪枝 CUDA 时间约 107.26→93.57 ms | A100 CUDA time + FLOPs |
| FasterVLM | Video-LLaVA，8 帧 | 2048 | 130 / 65 | 保留 130 时约保持 93.76% 相对性能 | 视频高 Token 场景 |
| DivPrune | LLaVA-1.5 单图 | 576 | 多种预算 | 报告端到端延迟和显存下降 | GPU 实测，选择算法本身复杂度较高 |
| DivPrune | LLaVA-1.6 单图 | LLaVA-1.5 的 3-5 倍，约 1728-2880 | 多种预算 | 高分辨率场景收益更明显 | GPU 实测 |
| DivPrune | LLaVA-NeXT-Video，8 帧 | 1152 | 多种预算 | 视频场景延迟下降 | GPU 实测 |
| 本项目 | Qwen3.5-2B，MMBench EN50 | **平均 95.28** | 50% 剪枝后平均约 47.64 | 当前实验没有稳定加速，准确率下降 | Apple MPS 端到端 benchmark |

### 主要出处

1. [FastV 原文](https://arxiv.org/abs/2403.06764)：LLaVA-1.5 的 336×336 图像产生 `576` Token，Video-LLaVA 使用 `2048` Token；论文强调视觉 Token 在样本输入中平均占约 `64%`。
2. [PyramidDrop 原文](https://arxiv.org/abs/2410.17247)：LLaVA-1.5 使用 `576` Token，LLaVA-NeXT 最高使用 `2880` Token；默认将 32 层分为 4 个阶段并逐阶段保留约一半 Token。
3. [SparseVLM 原文](https://arxiv.org/abs/2410.04417)：主要 LLaVA 实验从数百视觉 Token 开始，报告在 A100 上的 CUDA latency、FLOPs 和 cache memory。
4. [VisionZip 原文](https://arxiv.org/abs/2412.04467)：明确比较 LLaVA-1.5 的 `576→192/128/64` 与 LLaVA-NeXT 的 `2880→640/320/160`。
5. [FasterVLM 原文](https://arxiv.org/abs/2412.01818)：LLaVA-1.5 从 `576` Token 开始；Video-LLaVA 使用 `2048` Token；表格同时报告 Token 数、FLOPs、CUDA time 和准确率。
6. [DivPrune 原文](https://arxiv.org/abs/2503.02175)：LLaVA-1.5 使用 `576` Token，LLaVA-1.6 是其 3-5 倍，8 帧 LLaVA-NeXT-Video 使用 `1152` Token。
7. [Qwen3.5-2B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-2B)：语言模型共 24 层，采用 18 层 Gated DeltaNet 与 6 层 Gated Attention 的混合结构，不是这些论文主要测试的 32 层标准全注意力 LLaMA。

## 规模差异

以我们的平均值 `95.28` 为基准：

| 对比对象 | 原始视觉 Token | 是本项目的多少倍 | 本项目相当于其多少 |
|---|---:|---:|---:|
| Qwen3.5-2B 本项目 | 95.28 | 1.00× | 100.00% |
| LLaVA-1.5 | 576 | 6.05× | 16.54% |
| DivPrune 视频设置 | 1152 | 12.09× | 8.27% |
| Video-LLaVA | 2048 | 21.49× | 4.65% |
| LLaVA-NeXT | 2880 | 30.23× | 3.31% |

最直观的比较是：VisionZip 把 LLaVA-1.5 从 `576` 压缩到 `64` 时，已经属于只保留约 `11%` Token 的激进设置；而我们的原始输入平均只有 `95.28`，本身就非常接近论文的压缩后区间。若我们再裁掉 50%，平均只剩约 `48` 个 Token，已经比 VisionZip 的激进 `64` Token 配置更低。

## 为什么论文中的显著加速不容易复现

### 1. 可删除的绝对 Token 数不同

同样采用 50% 剪枝：

| 原始设置 | 删除的视觉 Token 数 |
|---|---:|
| 本项目平均 95.28 | 约 47.64 |
| LLaVA-1.5 576 | 288 |
| Video-LLaVA 2048 | 1024 |
| LLaVA-NeXT 2880 | 1440 |

排序、`topk`、索引重排、mask 更新和 KV cache 维护存在固定成本。论文模型一次能删除数百到上千 Token，而我们通常只能删除几十个，固定成本更难被后续层节省的计算抵消。

### 2. Qwen3.5-2B 已经不是论文假设的标准全注意力模型

FastV、PyramidDrop、VisionZip 和 FasterVLM 的主要收益来自缩短标准 LLaMA/Vicuna 解码器中每一层的注意力和 FFN 输入。Qwen3.5-2B 使用混合结构，24 层中只有 6 层 Gated Attention，其余 18 层是 Gated DeltaNet 线性注意力。

因此，即使视觉序列进一步缩短，它能减少的二次复杂度 attention 工作也比 32 层全注意力 LLaVA 少得多。论文 FLOPs 百分比不能直接迁移到 Qwen3.5-2B。

### 3. 我们的数据包含细粒度视觉任务

MMBench 包含 OCR、表格、局部目标和空间关系。我们的回归样本已经表明，少量局部 Token 可能决定最终答案。现有论文也承认：

- PyramidDrop 在更激进设置下会降低 TextVQA、DocVQA；
- VisionZip 从 576 压到 64 时平均性能仍约下降 6%；
- FasterVLM 从 576 压到 29 时只保留约 88%-90% 的相对性能；
- FastV 的文本-视觉 attention 排名在后续研究中被证明存在 attention shift 和 dispersion。

当原始 Token 已经很少时，每个 Token 平均承担的信息更多，继续删除的风险通常更高。

### 4. “FLOPs 降低”不等于“端到端变快”

FastV 和 PyramidDrop 的主要推理指标是理论 FLOPs。它们的结论成立于高视觉 Token 数、标准 Transformer 和 GPU kernel 能充分利用短序列的条件下。

本项目最终关心的是 TTFT、端到端生成时间和吞吐率。我们已经观察到，理论上减少视觉 Token 后，排序与动态张量操作会使 TTFT 上升。因此讨论时必须区分：

- 理论图像相关 FLOPs；
- CUDA/MPS kernel 时间；
- 完整 TTFT；
- decode 吞吐率；
- 最终准确率。

## 当前实验与文献是否一致

一致。

我们的 FastV 风格 EN50 实验中，平均约从 `95` 个视觉 Token 删除 `24-48` 个，但没有形成稳定收益：

- 准确率从基线 `0.76` 降至 `0.72-0.74`；
- 大多数配置 TTFT 上升；
- 所有已测配置的 decode 吞吐率都低于基线；
- 唯一出现轻微 TTFT 改善的配置，准确率和吞吐率仍下降。

这并不证明视觉 Token 剪枝在任何模型上都无效；它说明该方向的收益高度依赖“原始视觉 Token 很多”这一前提，而 Qwen3.5-2B 当前任务中的原始 Token 预算已经接近其他论文的压缩后预算。

## 建议给小组的决策

### 建议结论

将视觉 Token 剪枝从“主要性能优化方向”调整为“已验证的研究性支线”，暂不继续投入大量时间复刻复杂算法。

### 保留的最低限度工作

1. 保留当前实现、计时数据和失败报告，作为竞赛论文中的负结果与架构分析。
2. 如需最终确认，只再做一次不启用 eager attention、固定开销很低的温和实验；若仍无收益，正式停止该路线。
3. 不再用论文的 FLOPs 降幅预测本项目的 TTFT 降幅。

### 更值得转移的方向

优先寻找不依赖大量视觉 Token 的优化：

1. Qwen3.5 混合架构自身的算子与 kernel 优化；
2. Gated DeltaNet / Gated Attention 的实现与设备适配；
3. 模型量化、权重加载与算子融合；
4. 解码阶段、采样和 MTP 等直接影响吞吐率的优化；
5. 与队友已经验证的无精度损失方案组合。

## 一句话版本

主流视觉 Token 剪枝论文通常从 `576-2880` 个 Token 开始，而我们的 Qwen3.5-2B 平均只有 `95.28` 个，已经接近论文激进剪枝后的 Token 预算；再剪只能省几十个 Token，却仍需承担排序、动态裁剪和 cache 维护成本，因此精度风险高、真实加速空间小。
