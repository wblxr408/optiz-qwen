# ToMe 论文复现与 Qwen3.5 视觉编码器适配计划

日期：2026-07-12

## 目标

本方向准备复现 ToMe（Token Merging），将相似视觉 Token 在视觉编码器内部逐层合并，从而减少后续视觉 Attention 和 MLP 的计算量。目标不是追求理论 FLOPs，而是在本地 Qwen3.5-2B、Apple MPS 和 MMBench 上得到可复现的真实耗时与准确率结果。

已有基线表明：Qwen3.5-2B 的视觉编码器平均耗时约 `543.29 ms`，占完整图文 forward 的 `40.37%`，占视觉路径增量的 `64.44%`。因此 ToMe 比语言模型中间层 FastV 更直接地作用于当前最大视觉瓶颈。

## 论文信息

- 论文：Daniel Bolya 等，*Token Merging: Your ViT But Faster*，ICLR 2023。
- [OpenReview 论文](https://openreview.net/forum?id=JroZRaRw7Eu)
- [arXiv 论文](https://arxiv.org/abs/2210.09461)
- [官方实现](https://github.com/facebookresearch/ToMe)

官方仓库已经在 2025 年归档，许可证为 CC BY-NC 4.0。本项目不直接复制其代码，而是依据论文方法为 Qwen3.5 编写小型、可测试的本地实现，并在报告中保留来源说明。

## ToMe 的核心方法

### 1. 合并而非删除

Token pruning 会直接丢弃被判定为不重要的 Token；ToMe 将相似 Token 聚合到一个代表 Token 中，因此背景和前景中的冗余信息都可以被压缩。论文和官方实现均强调，合并能够比单纯剪枝更好地保留信息。

### 2. Bipartite Soft Matching

论文将 Token 交替划分为集合 A 和 B：

1. 对特征归一化并计算 A 到 B 的余弦相似度；
2. 每个 A Token 找到最相似的 B Token；
3. 从这些候选边中选择得分最高的 `r` 条；
4. 将对应的 A Token 合并到 B Token，其余 Token 原样保留。

该算法每层减少固定数量 `r`，避免一般图匹配的高昂开销。论文使用 Attention Key 作为匹配特征，并将合并放在 Attention 之后、MLP 之前，使当前层 Attention 仍能观察完整输入，而当前层 MLP 和后续层处理更短序列。

### 3. Token Size 与加权合并

每个初始 Token 的 size 为 1。合并时根据 size 做加权平均，并累加 size，记录新 Token 代表的原始区域数量。论文还提出 proportional attention，可在 Attention logits 中加入 `log(size)`，补偿合并后一个 Token 代表更大区域的影响。

### 4. 论文结论的适用边界

论文在标准 ViT、ImageNet 和视频模型上报告了吞吐提升，并展示了免训练使用与训练后使用两种设置。但这些数字不能直接迁移到 Qwen3.5：模型结构、位置编码、视觉输出形式、设备和下游任务均不同。本项目只把论文作为算法依据，所有收益结论以本地 benchmark 为准。

## Qwen3.5-2B 的适配约束

本地模型配置和 Transformers 实现显示：

| 项目 | Qwen3.5-2B 视觉编码器 |
|---|---|
| 视觉层数 | 24 |
| Hidden size | 1024 |
| Attention heads | 16 |
| Patch size | 16 |
| Spatial merge size | 2 |
| 末端输出 | 每连续 4 个 Patch 经 `PatchMerger` 映射到语言模型维度 |
| 位置编码 | 每层 Attention 使用二维 RoPE |
| 批处理形式 | 多图/视频以 packed sequence 和 `cu_seqlens` 表示 |

这带来三个不能回避的问题。

### PatchMerger 约束

原论文可以任意减少 Token 数；Qwen3.5 的末端 `PatchMerger` 要求 Token 保持 4 的倍数，并依赖每组 2×2 Patch 的排列。任意 Patch 级合并会破坏这一结构。

首版适配因此以一个 2×2 merge unit 为最小合并单位：匹配时用一个 unit 的 4 个 Patch 聚合得到 metric；合并时把源 unit 的四个相对位置分别合并到目标 unit 的对应位置。这样每次减少 4 个视觉编码器 Token，末端 `PatchMerger` 仍能按原结构执行。

### RoPE 约束

标准 ViT 的绝对位置嵌入通常只在输入处加入；Qwen3.5 在每层视觉 Attention 中重复应用 RoPE。合并后的 Token 没有唯一的新坐标。

首版采用目标 Token 的位置：源 unit 合并进目标 unit 后，保留目标 unit 对应的四组 RoPE。它与 ToMe 的“将源聚合到目标”语义一致，也避免构造未经模型训练的新坐标。报告必须将其标记为 Qwen3.5 适配假设，并通过准确率实验验证。

### Packed Sequence 约束

匹配不得跨图片或视频帧发生。实现必须根据 `cu_seqlens` 对每个视觉样本独立匹配，并在每层合并后重新生成累计长度。首个正式版本不允许只支持单图后静默处理其他输入；不满足结构约束时应立即报错。

## 实施计划

### 阶段 1：建立可验证的 ToMe 核心算子

新增纯 PyTorch 模块，实现：

- unit 级 bipartite soft matching；
- size-weighted merge；
- 源/目标索引和 Token 数统计；
- 单图与 packed 多图边界隔离；
- CPU 与 MPS 一致的确定性行为。

单元测试应覆盖：`r=0` 恒等、Token 数变化、size 守恒、不能跨图合并、输出保持 4-Token unit 结构、非法 `r` 立即报错。

完成证据：测试输出、一个小张量手工算例，以及合并前后 shape/size 记录。此阶段不加载完整模型，不宣称性能收益。

### 阶段 2：接入单个视觉 Block

按照论文位置，将合并插入指定 block 的 Attention 残差之后、MLP 之前：

```text
x = x + Attention(Norm(x))
x, size, position, cu_seqlens = ToMe(x, metric, ...)
x = x + MLP(Norm(x))
```

匹配 metric 优先使用该层 Attention 的 Key。为避免重复执行 QKV 投影，需要让视觉 Attention 显式返回本次已计算的 Key metric，而不是在 ToMe 中重新计算投影。

首轮只在一个中间层启用，并提供明确配置：启用状态、层号和每层减少的 unit 数。默认行为必须保持原模型不变。

完成证据：baseline 输出逐位不变；启用后模型可完成一次真实 MPS 推理；日志能显示每层输入 Token、输出 Token 和合并耗时。

### 阶段 3：扩展为多层 schedule

支持 24 层视觉编码器的逐层 `r` schedule，但不一开始搜索大量组合。先比较三类简单方案：

| 方案 | 合并层 | 用途 |
|---|---|---|
| Late | 16-20 层 | 最低精度风险，验证链路 |
| Middle | 8-16 层 | 平衡可节省计算与语义稳定性 |
| Progressive | 从浅到深逐步增加或保持固定 `r` | 接近论文逐层合并思路 |

默认不在最浅层立即激进合并，因为 OCR、局部目标和空间关系可能依赖低层细节。每层必须保证至少保留预设的最小 unit 数，`r` 超界直接报错。

完成证据：逐层 Token 曲线、逐层 Attention/MLP/merge 耗时，以及输出给语言模型的视觉 Token 数。

### 阶段 4：小规模可行性筛选

使用固定 EN10 样本，先排除明显无效配置。每个配置记录：

- 视觉编码器时间；
- ToMe 匹配和 merge 时间；
- 完整 TTFT；
- 输出视觉 Token 数；
- 正确题数及变化题目；
- MPS 峰值内存（若当前 PyTorch 接口可稳定测量）。

可行性门槛：视觉编码器净耗时必须下降；ToMe 固定开销不能吃掉后续层节省；不得出现运行错误或非有限值。未满足门槛的 schedule 不进入大样本评估。

### 阶段 5：完整准确率与性能评估

对通过筛选的 2-3 个配置运行至少 EN50；若方案稳定，再补 CN50。所有候选必须使用同一模型、数据顺序、Prompt、生成参数、预热方式和答案解析逻辑。

最终比较：

| 指标 | 判断作用 |
|---|---|
| 准确率与相对准确率 | 判断信息损失 |
| 视觉编码器耗时 | 验证是否击中目标瓶颈 |
| ToMe 自身耗时 | 判断算法固定成本 |
| TTFT | 竞赛主要性能收益 |
| 总耗时和吞吐率 | 检查端到端价值 |
| 每层 Token 数 | 解释收益来源 |
| 逐题答案变化 | 定位 OCR、空间和细粒度风险 |

结果图应至少包含速度-准确率 Pareto 图和逐层 Token/耗时曲线。汇总指标用表格，不为少量数字单独作图。

## 消融实验

只有基础实现形成正向结果后，才依次进行以下消融，避免过早增加分支：

1. Attention Key metric 与 hidden-state metric；
2. 普通加权平均与 proportional attention；
3. Late、Middle 和 Progressive schedule；
4. 固定 `r` 与按初始 unit 比例确定 `r`；
5. 目标位置 RoPE 与其他通用位置策略。

不进行题型识别、OCR 特判或针对 MMBench 样本的规则优化。

## 工程边界

- ToMe 默认关闭，baseline 行为不变。
- 实现只依赖当前 PyTorch，不引入 Triton、CUDA 或外部 ToMe 包。
- Mac MPS 负责正确性、准确率和第一轮真实性能验证。
- 后续服务器只复核硬件相关收益，不作为本地开发的前置条件。
- 配置必须进入 benchmark 元数据，确保每份结果可复现。
- 不提供静默 fallback；结构、shape、schedule 或设备不满足要求时立即报错。
- 每个关键阶段先提供代码、测试、日志和图表审查，得到确认后再提交。

## 停止条件

出现以下任一情况，应停止继续堆叠 ToMe 复杂度并保留负结果：

1. 中等合并比例下视觉编码器没有稳定净加速；
2. merge 开销持续高于 Attention/MLP 节省；
3. 在 EN50 上准确率下降明显，但 TTFT 改善很小；
4. unit 级合并受 PatchMerger 和 RoPE 约束，无法形成有效压缩；
5. 收益只存在于个别样本或依赖数据集特定规则。

## 计划提交节点

1. ToMe 核心算子与单元测试；
2. Qwen3.5 单层接入、开关与运行日志；
3. 多层 schedule、细粒度 profiler 与小样本筛选报告；
4. EN50/CN50 完整 benchmark 与最终决策报告。

每个节点在提交前由组内审查。当前文档只定义复现方案，不代表已经实现或证明 ToMe 有效。
