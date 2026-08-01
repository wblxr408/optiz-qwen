# Qwen3.5-2B 推理优化路线综述与项目决策

日期：2026-08-01

> PPU 实测已经完成，详见
> [`ppu_gdn_server_validation_20260801.md`](ppu_gdn_server_validation_20260801.md)。模块画像确认
> GDN 占已计时语言模块 prefill 的约 86.8%、decode 的约 70.0%；仅 decode 四投影融合在
> 中英文共 300 题中保持最终选项一致，吞吐分别提高 1.28% 和 3.01%。

## 1. 为什么重新做综述

本项目已经将 FastV 式剪枝、ToMe、PiToMe 和 DToMe 适配到 Qwen3.5-2B，并在
Apple MPS、NVIDIA GPU 和目标 PPU 上做了分层及端到端测试。最新 PPU 证据是：

- L8/R32 将视觉编码器耗时降低 6.86%；
- 但公开英文 50 题的配对准确率从 74% 降至 68%；
- TTFT 从 150.309 ms 变为 150.710 ms，吞吐从 70.937 变为 70.603 tok/s。

这说明局部算子或 FLOPs 改善不会自动转化为竞赛指标改善。本综述因此不按论文
中的最大加速比排名，而是按下列条件评估：

1. 是否改善赛事实际的 TTFT 或单样本 decode 吞吐；
2. 是否适用于 2B 模型、短上下文、短回答和单请求；
3. 是否匹配 Qwen3.5 的 3:1 Gated DeltaNet / Full Attention 混合架构；
4. 是否能在 PPU-ZW810E 上获得真实内核加速；
5. 是否保持官方评测逻辑和模型输出质量。

## 2. 总览

| 方向 | 论文中的典型收益 | 对当前任务的判断 | 精度风险 | PPU 工作量 | 优先级 |
|---|---|---|---|---|---:|
| GDN 快路与融合内核 | FlashQLA 的 GDN prefill 单算子 2–3× | 直击 75% 语言层，与本地实测一致 | 低 | 高 | **S** |
| 计算图捕获与轻量融合 | 减少 kernel launch 和中间读写 | 短序列、2B、batch=1 更受调度开销影响 | 极低 | 中高 | **S** |
| W8A8 / W4A8 与融合 GEMM | SmoothQuant 最高 1.56×，QServe 服务吞吐 1.2–3.5× | 必须有 PPU 低比特内核；仅压权重不会加速 | 低至中 | 高 | **A** |
| 视觉预处理 token 预算 | 直接减少视觉编码和 prefill | 实现简单，但 OCR/局部定位风险高 | 中高 | 低 | **A-** |
| Full Attention / SDPA 内核 | FlashAttention-2 相对旧 attention 2–4× | 只覆盖 25% 语言层，序列也较短 | 极低 | 中高 | **B+** |
| 视觉编码器算子融合 | FastVLM 显示高效 encoder 能改善延迟 | 不替换权重时仍可做 norm/MLP/attention 融合 | 低 | 高 | **B** |
| 视觉 token 剪枝/合并 | 高 token LLaVA 中常报 37% latency 或 2× GPU 时间 | 原 token 已少，PPU 端到端失败 | 高 | 中 | **C** |
| KV Cache 量化/稀疏化 | KIVI 在长上下文高并发中 2.35–3.47× 吞吐 | 本任务 cache 短，且仅 6 层 Full Attention | 低至中 | 高 | **C** |
| 推测解码 / Medusa / EAGLE | 大模型中常见 2–3.6× | 2B 目标验证便宜，草稿开销与 GDN 状态回滚困难 | 低（正确验证时） | 极高 | **C-** |
| 结构化层剪枝 / 早退出 | 可直接减少层数 | 2B 模型冗余少，通常需重训 | 高 | 中 | **D** |
| 服务调度、连续 batching | 高并发服务吞吐可超过 2× | 若赛事按单请求 TTFT/吞吐评分，基本不适用 | 无 | 高 | **D** |

## 3. 最值得转向的主线：Gated DeltaNet 内核

Qwen3.5 的语言模型每 4 层中有 3 层 Gated DeltaNet，只有 1 层 Full Attention。
Transformers 官方文档明确提示，缺少 `fla` 和 `causal_conv1d` 时会回退到更慢、
更耗内存的 PyTorch 路径。学校 PPU 当前运行时正在走这条 fallback。

最相关的外部证据是：

- [Gated DeltaNet 论文](https://arxiv.org/abs/2412.06464) 为门控 delta rule 给出了硬件友好的并行形式；
- [Flash Linear Attention](https://arxiv.org/abs/2312.06635) 强调线性注意力只有配合 IO-aware 内核才能
  实现理论优势；
- Qwen 官方 [FlashQLA](https://github.com/QwenLM/FlashQLA) 对 GDN chunked prefill 融合和代数重写，
  在 NVIDIA Hopper 上报告相对 FLA Triton 的 2–3× forward 加速；
- SGLang 的 [Qwen3.5 优化跟踪](https://github.com/sgl-project/sglang/issues/18590) 将 GDN prefill、
  decode、state layout 和图捕获列为核心优化项；
- 阿里云 APG 的 Qwen3.5 实践也将收益归因于 GDN 门控、状态更新和输出投影融合，
  以及静态计算图重放。

本仓库的 NVIDIA 结果与这一文献线索高度一致：GDN CUDA 快路将全量平均 TTFT
从 293.287 ms 降至 52.127 ms，吞吐从 132.342 提高到 144.161 tok/s，4317 题准确率
只变化 -0.0695 个百分点。这不能直接代表 PPU，但它是目前唯一同时具备“论文/官方
实现支持”和“本项目大样本实测支持”的高收益路线。

### 对 PPU 的正确目标

不应尝试在 PPU 上安装 NVIDIA FLA 或原样搬运 FlashQLA。应该复用其算法结构，通过
PPU SDK 实现或调用下列能力：

1. 将 `causal_conv1d`、gate、state update 和 output projection 中的可融合部分合并；
2. 使 recurrent state 保持在设备端友好布局，避免每 token 重复转置与 HBM 往返；
3. 分别优化 prefill 的 chunkwise 路径和 decode 的 recurrent 单 token 路径；
4. 对固定 decode shape 做图捕获或调度重放；
5. 与 FP32 参考实现做逐层数值对齐，禁止静默 fallback。

## 4. 量化：算法不是瓶颈，内核才是

[AWQ](https://arxiv.org/abs/2306.00978) 保护少量显著通道，适合 W4A16 权重压缩；
[SmoothQuant](https://arxiv.org/abs/2211.10438) 将 activation outlier 的量化难度迁移到权重，
实现 W8A8，论文报告最高 1.56× 加速与 2× 内存下降。
[QServe](https://arxiv.org/abs/2405.04532) 则表明 W4A8KV4 需要重排布局、寄存器级并行和
融合反量化，否则反量化本身可产生 20%–90% 开销。

本项目的 AWQ 已经完美复现了这个边界：模型目录缩小 34.92%，但 Transformers 在
运行时解包权重，最终 TTFT +0.95%、吞吐 -3.05%。因此：

- 继续调 AWQ 校准参数不是性能主线；
- 先查 PPU 是否有原生 INT8 / INT4 GEMM、支持的分组大小和权重布局；
- 若只支持 INT8，优先 W8A8，因为它能同时加速 prefill 的 GEMM；
- 若支持 INT4，必须将 dequant 融合进 GEMM/GEMV，不允许先恢复 BF16 大矩阵；
- 视觉编码器与 GDN 层需分别做敏感性校准，不应统一粗暴量化。

## 5. 视觉方向：从“删 token”转向“算子与预处理”

### 5.1 为什么论文的大数字不适用

- [FastV](https://arxiv.org/abs/2403.06764) 主要从 576 或 2048 个视觉 token 开始；
- [SparseVLM](https://arxiv.org/abs/2410.04417) 在 LLaVA 上报告 54% FLOPs 下降、
  37% CUDA latency 下降和约 97% 相对准确率；
- [VisionZip](https://arxiv.org/abs/2412.04467) 在 LLaVA-NeXT 的 2880 token 高分辨率设置中
  报告最高 8× prefill 和约 2× GPU inference；
- [PyramidDrop](https://arxiv.org/abs/2410.17247) 利用深层视觉冗余逐阶段减少 token。

它们的共同前提是“视觉 token 很多且后续是多层全注意力”。Qwen3.5-2B 在本项目的
语言模型输入平均仅约 95 个视觉 token，且 75% 语言层是 GDN。这就是我们实测只得到
局部编码器收益，却无端到端收益的根本原因。

### 5.2 仍值得保留的视觉实验

1. **视觉编码器内核融合。** 保留原权重，优化 QKV、SDPA、MLP、norm 和 PatchMerger，
   可以无精度损失，但需 PPU 算子支持。
2. **官方 processor 的视觉像素预算。** 实现成本最低，但必须在 OCR、细粒度定位、
   图表和空间关系类别上做全量回归。现有 10 题的 192×192 结果不足以成为决策证据。
3. **更高效视觉编码器。** [FastVLM](https://arxiv.org/abs/2412.13303) 说明 encoder 架构对
   延迟影响很大，但替换 encoder 会改变权重与模型契约，通常需要重训，不适合当前赛事阶段。

## 6. KV Cache：好论文，错场景

[KIVI](https://arxiv.org/abs/2402.02750) 使用 2-bit 非对称 KV 量化，在长上下文、高并发
负载中以 2.6× 较小峰值内存换来 2.35–3.47× 吞吐。QServe 将 KV4 收益和融合
attention 内核绑定，同样针对服务吞吐。

当前任务恰好缺少它们的成立条件：

- 公开样本输入只有数百 token，输出通常仅数个到十几个 token；
- Qwen3.5-2B 只有 6 层 Full Attention 使用传统 KV cache；
- batch=1 时不存在“省 cache 后扩大 batch”的吞吐收益；
- quantize/dequantize、pack/unpack 和动态 cache 管理开销反而会主导。

仓库的 KIVI 和 QServe KV 结果没有赢过 baseline，与文献的适用边界一致。除非后续评测
改为长上下文或多并发，否则不应再将 KV 压缩作为短期主线。

## 7. 推测解码、层剪枝和调度

### 7.1 推测解码

[EAGLE](https://arxiv.org/abs/2401.15077) 在 LLaMA2-Chat 70B 上报告 2.7–3.5× latency 加速；
[Medusa](https://arxiv.org/abs/2401.10774) 报告 2.2–3.6×，但需训练额外预测头或联合微调。
这类方法通过目标模型严格验证时可保持分布，但对当前项目不划算：

- 2B 目标模型本身单次验证便宜，草稿模型/额外头开销占比更高；
- 回答较短，可节省的解码轮数有限；
- Qwen3.5 的 GDN 包含 recurrent state，接受/拒绝草稿 token 需要正确回滚或重放状态；
- 模型未随权重提供可直接使用的 Medusa/EAGLE 头。

### 7.2 层剪枝与早退出

[ShortGPT](https://arxiv.org/abs/2403.03853) 通过 Block Influence 删除冗余层；
[LayerSkip](https://arxiv.org/abs/2404.16710) 使用 layer dropout 和 early-exit loss 训练模型，
然后进行自推测解码。官方实践特别指出，未经 LayerSkip 训练的普通权重往往比自回归
baseline 更慢，因为早层接受率低。

Qwen3.5-2B 已是小模型，直接删层的精度风险高于 7B–70B 模型，且混合 GDN/Attention
层不可按普通 Transformer 层同质处理。可以作为论文性尝试，但不应优先于无损内核
优化。

### 7.3 调度与 batching

[DeepSpeed-FastGen](https://arxiv.org/abs/2401.08671) 通过 Dynamic SplitFuse 报告最高 2.3× 有效
吞吐和约 2× 平均延迟改善，但它解决的是多请求服务问题。如果赛事按单样本串行评测，
连续 batching、paged KV 和 prefill/decode 分离不会自动改善单请求 TTFT，甚至会增加队列开销。

## 8. 推荐的新工作顺序

### 阶段 0：用 PPU profiler 重建瓶颈图

目标不是再测一个总 TTFT，而是将时间分为：

1. image preprocessing 和 H2D；
2. vision encoder；
3. projector / PatchMerger；
4. GDN prefill；
5. Full Attention prefill；
6. MLP / norm / projection；
7. 首 token 后的 GDN recurrent decode；
8. Full Attention decode、sampler 和 Python/streamer 调度。

只有这份剖析能决定 GDN 中应先优化 prefill 还是 decode，以及图捕获的收益上限。

### 阶段 1：先查设备已有能力

1. 搜索 PPU SDK、ModelZoo 和预编译库中的 GDN / DeltaNet / linear attention 算子；
2. 检查 PPU 计算图捕获、静态图、算子融合与自定义算子接口；
3. 列出 PPU 原生 BF16、FP16、INT8、INT4 GEMM/GEMV 支持的 shape 和 layout；
4. 检查 Transformers fallback 中是否存在多余 transpose、contiguous、CPU 回读或小算子链。

这一阶段不需要上传团队完整代码，也比盲目手写 PPU kernel 风险低。

### 阶段 2：建立两个最小原型

1. **GDN decode 图捕获/融合原型**：数值完全一致，重点减少 launch 和 state 往返。
2. **GDN prefill chunk 原型**：参考 FLA/FlashQLA 的分块并行形式，但使用 PPU 支持的算子和
   布局。

两者必须先在单层上与 FP32 参考对齐，再接入 18 层 GDN，最后跑完整 benchmark。

### 阶段 3：备选项

- 若 PPU 有高性能 INT8 GEMM，开始 SmoothQuant/W8A8 小规模校准；
- 若图捕获成熟，将固定 shape 的 decode 步骤整体捕获；
- 若 GDN 无法自定义优化，再做 Full Attention SDPA 和视觉 encoder 融合；
- 视觉像素预算只作为可选的速度–精度对照，不与无损优化混合评价。

## 9. 停止投入的方向

1. 不再围绕 ToMe/PiToMe/DToMe 调层号、阈值或匹配分数，除非有新架构性思路。
2. 不在没有融合低比特内核的情况下继续调 AWQ 权重。
3. 不对当前短上下文继续调 KIVI/QServe KV 分组大小。
4. 不使用论文 FLOPs 数字替代端到端 TTFT、吞吐和准确率。
5. 不优先复现需要额外训练头或新模型权重的推测解码、LayerSkip 或 FastVLM。

## 10. 最终决策

本轮综述支持将项目的主要技术问题从“删除哪些 token”改为：

> **如何使 Qwen3.5-2B 的 Gated DeltaNet 在 PPU 上不再走 PyTorch fallback，并将其门控、
> 状态更新和投影重写为 PPU 友好的融合执行路径？**

这条路线不需要牺牲视觉信息，同时作用于 18/24 语言层，又已经得到官方内核工作
与本项目 NVIDIA 大样本结果的双重支持。它实现困难，但与小幅调整 ToMe 相比，具有
数量级更大的潜在收益，也更符合赛题对算子优化和软硬协同的要求。

## 11. 主要参考资料

1. [Qwen3.5 Transformers 架构文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5)
2. [Gated Delta Networks](https://arxiv.org/abs/2412.06464)
3. [Gated Linear Attention / Flash Linear Attention](https://arxiv.org/abs/2312.06635)
4. [QwenLM FlashQLA](https://github.com/QwenLM/FlashQLA)
5. [FlashAttention-2](https://arxiv.org/abs/2307.08691)
6. [AWQ](https://arxiv.org/abs/2306.00978)
7. [SmoothQuant](https://arxiv.org/abs/2211.10438)
8. [QServe](https://arxiv.org/abs/2405.04532)
9. [KIVI](https://arxiv.org/abs/2402.02750)
10. [FastV](https://arxiv.org/abs/2403.06764)
11. [SparseVLM](https://arxiv.org/abs/2410.04417)
12. [VisionZip](https://arxiv.org/abs/2412.04467)
13. [PyramidDrop](https://arxiv.org/abs/2410.17247)
14. [FastVLM](https://arxiv.org/abs/2412.13303)
15. [EAGLE](https://arxiv.org/abs/2401.15077)
16. [Medusa](https://arxiv.org/abs/2401.10774)
17. [LayerSkip](https://arxiv.org/abs/2404.16710)
18. [ShortGPT](https://arxiv.org/abs/2403.03853)
19. [DeepSpeed-FastGen](https://arxiv.org/abs/2401.08671)
20. [SGLang Qwen3.5 优化跟踪](https://github.com/sgl-project/sglang/issues/18590)
