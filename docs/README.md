# Qwen3.5-2B VLM 边缘部署优化方案总览

## 赛题目标

根据 [赛题.md](/D:/VLM/赛题.md:1)，任务核心是：

- 基于 `Qwen3.5-2B` 完成 VLM 部署
- 在 `PPU` 服务器上做推理优化
- 同时覆盖模型压缩、系统调度、算子融合、内存管理、硬件适配
- 输出真实源代码、性能对比结果和技术报告

## 当前基线入口

主办方公开给出的 DNDX 选手资料已并入当前仓库，基线评测入口收敛为：

- 根目录 `benchmark_public.py`：兼容主办方自测命令
- 根目录 `evaluation_wrapper.py`：兼容主办方提交接口
- `src/optiz_qwen/evaluation/dndx_public_benchmark.py`：仓库内维护实现
- `src/optiz_qwen/evaluation/dndx_wrapper.py`：仓库内维护实现

公开自测资源路径统一为：

- 数据集：`resources/eval_dataset/raw/mmbench_public/`
- 模型目录：`resources/model_weights/raw/Qwen3.5-2B/`
- 输出结果：`benchmarks/output/`

## 瓶颈认知修正（2026-08，实测结论）

**本项目此前的优化重心是错的。** 早期路线假设瓶颈在显存带宽（HBM bandwidth），
因此把权重量化和 KV 量化放在主线。在 PPU-ZW810E 上实测后结论必须改为：

> **瓶颈是 CPU 侧的 kernel 派发（dispatch）开销，不是 HBM 带宽。**

target 实测证据（详见 [ppu_optimization_design.md](ppu_optimization_design.md)）：

| 指标 | 实测值 |
| --- | --- |
| 每 token CUDA kernel 数 | 5778 |
| 单 decode step 实测耗时 | 19.3 ~ 21.1 ms（47 ~ 51 tok/s） |
| 4.426 GB 权重的 HBM 物理下限 | 2.20 ms/step（454 tok/s） |
| 实测 / 物理下限 | **8.9x** |
| 连续排 20 step 的 CPU 下发耗时 | 20.656 ms/step vs 20.678 ms 墙钟 |

最后一行是判定性的：CPU 下发时间几乎等于墙钟时间，设备是在等 CPU，而不是在等
显存。decode 的 GEMV 已经跑到 1521 ~ 2073 GB/s，而 roofline 是 2011 GB/s，
带宽侧**没有可回收的余量**。

因此优化重心改为：**消除逐 token 的 kernel 派发**（CUDA Graph 捕获重放）
＋ **按阶段选择注意力后端**（prefill 用 sdpa、decode 用 flash_attention_2）。

### 补充实测：prefill 同样是 dispatch-bound

用 `scripts/profile_ppu_prefill.py` 在 target 上测 prefill（340 token prompt，5 次排队
forward 不同步）：

| 指标 | 实测值 |
| --- | --- |
| `cpu_issue_fraction`（CPU 下发时间 ÷ 墙钟） | **0.9882 ~ 0.9893** |
| 判定 | **dispatch-bound**，每个样本都是 |
| 单次 prefill kernel launch | **2423 次**（此前写的 4700 是重复计数，见下） |
| 单次 prefill device 时间 | **27.3 ms**（墙钟 53 ~ 59 ms，`device_busy_fraction` 0.47 ~ 0.51） |

> **更正**：早先的"161 种 / 4700 次调用"和"top-12 device op 合计 42.5 ms"都来自把
> `prof.key_averages()` 的全部行相加。该接口同时返回 CPU 算子行（`aten::mm`）和 device
> kernel 行（`gemm_ktype0_...`），而算子行会把自己子节点的 device 时间和调用次数计入自身，
> 所以整表求和是把同一份工作数了两遍。真实值是 **2423 次 kernel launch / 13698 次算子调用
> / 27.3 ms device 时间**（`scripts/profile_ppu_prefill_headroom.py`）。42.5-of-53 这个读数
> 意味着 device 有 ~80% 时间在忙，与上面一行的 dispatch-bound 判定直接矛盾——这个矛盾就是
> 线索，最后靠重测而不是靠挑一个顺眼的数字解决。

这条结论有两个直接后果。一是 `causal_conv1d` 装上之后 TTFT 几乎不动（见下），因为它
减少的是 device 工作量，而这条路径并没有在等 device。二是 TTFT 剩下的大杠杆是
**把 prefill 也捕获/编译掉**——这一项现已实测完毕，结论是**上限只有 ~2x 且两条路都不通**，
见下节。

唯一属于"纯浪费"而非派发问题的是 lm_head：greedy prefill 只读 `logits[:, -1, :]`，
但模型默认对全部 ~340 个 position 做 vocab 248320 的投影。实测 `lm_head(hidden)`
3.017 ~ 3.040 ms，`lm_head(hidden[:, -1:, :])` 0.472 ~ 0.482 ms，浪费 **2.55 ms**。
传 `logits_to_keep=1` 即可裁掉，首 token 由构造保证不变（下游取的是同一行）。

### prefill 捕获/编译：已实测，结论是此路到顶

四组实测，全部在 target 上跑，详见 `docs/ppu_optimization_design.md` 第 2 节 stage 6。

**一、上限是 ~2x，不是 decode 的 8.9x。** `device_busy_fraction` 0.47 ~ 0.51，即墙钟里约
一半是 device 空转。把派发开销全部消掉，prefill 也只能从 53 ~ 59 ms 降到 ~27 ms，
折合 **1.95 ~ 2.13x**。decode 那种量级的余量在 prefill 上不存在。

**二、0.99 的 `cpu_issue_fraction` 有两种解释，而它是贵的那一种。** 排队不同步只能看出
"CPU 一直在忙"，分不清是 *下发不过来*（要靠捕获）还是 *被同步卡住*（把值留在 host 即可）。
用 `torch.cuda.set_sync_debug_mode("warn")` 逐行归因：单次 prefill **93 ~ 94 次 host 同步，
其中 72 次来自同一行** —— `modeling_qwen3_5.py:968`，视觉注意力里的 `lengths.tolist()`
（每块 3 次 D2H × 24 块，而且每次算出的都是同一个 list）。

**三、把这 72 次同步去掉，数值逐位一致，但只值 ~1%。** `kernels/vision_prefill_sync.py`
从 `grid_thw` 在 host 侧闭式推出分块长度（与上游 `repeat_interleave(h*w, t).cumsum()`
一致），三个样本 94→22 / 93→22 / 93→22，prefill p50 提升 **+1.41% / +1.01% / +1.05%**，
`max_abs_logit_delta` 全为 **0.0**。去掉 77% 的同步只换来 1%，`cpu_issue_fraction` 依旧
≈0.99 —— 这就证明 prefill 是**下发受限**而非被同步卡住：CPU 的开销摊在 13698 次算子调用上，
不是几个卡点，只有整体捕获或编译才动得了它。默认关闭，开关
`OPTIZ_QWEN_VISION_SYNC_ELISION`。

**四、shape 不固定，捕获无从下手。** CUDA Graph 捕获要求 shape 固定。语言侧做不到：50 个
样本有 **46 种不同 prompt 长度**（137 ~ 363）。视觉塔的 shape 来自 `image_grid_thw` 而不是
prompt，值得单独查，结果同样不行：**24 种 `image_grid_thw`、18 种 `pixel_values` shape**，
最高频的 `[1,16,24]` 也只占 9/50。

**五、`torch.compile(dynamic=True)` 反而更慢，而且数值不一致。** 这是唯一能在不固定 shape
的前提下削减框架开销的手段（D1 当初否掉 `torch.compile` 针对的是 `reduce-overhead` 的
cudagraph trees，不覆盖这个配置，所以值得单独测）。实测语言栈 **-12.7% ~ -14.3%**、
视觉塔 **-12.1% ~ -13.0%**：语言侧撞到 `recompile_limit (8)`（`attention_mask` dtype
Long vs Bool 反复触发重编译），视觉侧在 `modeling_qwen3_5.py:989` 的数据依赖 split 上
graph break。同时 `max_abs_logit_delta` 达 0.328 ~ 0.5，两个 arm 各有 1/3 样本首 token 翻转。
**否决。**

结论：prefill TTFT 在当前层面已到顶。再往下要么做 shape 分桶 + padding 捕获（未测，
padding 的精度代价也未测），要么进 stage 7 原生内核——而后者优化的是 device 时间，
正好是墙钟里**不构成瓶颈**的那一半。

### `causal_conv1d` 已编译可用，但不算加速项

源码 wheel 在 target 上编译成功（`causal_conv1d-1.6.2.post1`，约 13 分钟），
`is_fast_path_available` 现在为 **True**，kernel 数值正确（bf16 下与 `F.conv1d`+SiLU
参考实现最大绝对误差 0.0312）。但它只减少了 **2423 次 kernel launch 中的 72 次**，
prefill 全量 forward 54.926 → 53.577 ms，判定不变。它的价值是关闭了一个未验证项、
消除了 transformers 的 fast-path 警告，并且这个"无效果"本身就是 prefill
dispatch-bound 的证据。**不要把它写成 GDN prefill 的性能收益。**

## 技术路线（已按实测修正）

当前主线，全部为 dispatch 侧优化：

- 运行时调度：`Prefill / Decode` 分离
- **CUDA Graph decode**：捕获一次 decode step 后逐 token 重放，替代每 token 5778 次
  kernel 派发（`scheduling/cuda_graph_decode.py`）
- **分阶段注意力后端**：prefill 用 `sdpa`、decode 用 `flash_attention_2`；
  在 FA2 下捕获图、捕获后把全局 config 翻回 sdpa（`kernels/attention_backend.py`）
- **`fla-core` Triton gated-delta-net**：prefill 108.64 → 54.25 ms，逐字节一致
- **prefill lm_head 只投影最后一个 position**（`logits_to_keep=1`）：50 样本实测在
  CUDA Graph 之上再拿 **+5.10% TTFT**，答案一致性 50/50
  （`scheduling/prefill_decode.py`，默认开启，`OPTIZ_QWEN_PREFILL_LAST_LOGIT_ONLY=0` 关闭）
- PPU 适配：算子兼容性扫描、长尾算子补齐

### 量化路线的重新定位

| 方案 | PPU 实测结果 | 新定位 |
| --- | --- | --- |
| INT4 KV chain（延迟 split packed-KV） | 吞吐 **-2.38%**，TTFT -1.82% | **显存优化备选**，不是加速方案 |
| W4A16 权重量化 | 在本硬件上未取得吞吐收益；`torch._int_mm` 仅 0.17 TOPS，torchao int8wo 0.23x / int4wo 0.25x of bf16 | **显存优化备选**，不是加速方案 |

两者都保留在仓库里、都保持可开关，但都不再算作性能加速主线。原因是同一条：
dispatch-bound 的负载下，减少每次访存的字节数不会减少 kernel 个数，反而
（KV chain 的情况）额外引入了 Triton kernel，把派发次数推高了。

显存优化备选（保留、可开关、不计入加速主线）：

- 视觉压缩：`FastV` 风格层间视觉 Token 剪枝
- 权重量化：`AWQ W4A16`
- KV 管理：分页 KV Cache、延迟 split packed-KV INT4 链

增强分支而非首发主线：

- `SparseVLM` 风格文本引导稀疏化
- `EAGLE` 风格投机解码

## 四层优化结构

现有 [vlm_optimization_full_stack_architecture.svg](/D:/VLM/vlm_optimization_full_stack_architecture.svg:1) 对应的四层结构保留不变，但含义进一步收敛为：

### 第一层：模型压缩

- `AWQ` 权重量化
- 视觉 Token 剪枝
- 必要时再引入 KV 量化和 VLM 专用校准

### 第二层：系统调度

- `Prefill / Decode` 分离执行
- 分页 KV 管理
- 请求状态与缓存布局优化

### 第三层：内核算子融合

- `Dequant + GEMM/GEMV` 融合
- Attention 核心路径融合
- FFN 路径融合
- RoPE / RMSNorm / SiLU 等轻量算子补充融合

### 第四层：PPU 硬件适配

- 算子支持性映射
- 不支持算子的等价重写或查表补齐
- 权重打包格式与带宽友好布局

## 实施顺序

1. 跑通基线并做 profiling
2. 接入 `AWQ`
3. 接入视觉 Token 剪枝
4. 做精度与 `TTFT` 参数扫描
5. 实现 `Prefill / Decode` 分离
6. 实现 KV 布局和主路径 Kernel 融合
7. 最后做 `PPU` 深度适配与增强项

> 第 7 步（PPU 原生内核开发）**尚未开始**。当前全部收益来自 dispatch 消除
> （CUDA Graph）、注意力后端选择、以及去掉 prefill 里多余的 lm_head 投影，
> 这些路径是在 PPU 硬件上通过其 CUDA 兼容运行时执行的，不含任何 PPU 原生自定义算子。

## 当前推荐提交版本

**PPU 混合方案 + prefill lm_head 裁剪**。50 样本 MMBench dev-en，
`max_new_tokens=256`，`max_cache_len=2048`，greedy，batch 1，每个 arm 独立进程，
**由仓库正式入口 `optiz_qwen.evaluation.dndx_public_benchmark` 产出**（不是探针脚本），
`fla-core==0.5.2` + `causal_conv1d==1.6.2.post1` 均已安装。证据在
`benchmarks/output/ppu_hybrid_trim_ab_50samples.json`，由
`tests/test_ppu_hybrid_trim_regression.py` 钉住。

三个 arm，目的是把两个 TTFT 杠杆拆开来看：

| 指标 | A baseline | B hybrid | C hybrid + prefill 裁剪 |
| --- | --- | --- | --- |
| 精度 | 0.76 (38/50) | 0.76 (38/50) | **0.76 (38/50)** |
| TTFT 均值 | 58.436 ms | 55.378 ms | **52.556 ms** |
| TTFT p50 | 57.615 ms | 54.816 ms | **51.870 ms** |
| 吞吐 | 44.439 tok/s | 162.937 tok/s | **162.312 tok/s** |
| TTFT vs A | — | +5.23% | **+10.06%** |
| 吞吐 vs A | — | +266.65% | **+265.25%** |

CUDA Graph 本身值 **+5.23%** TTFT，prefill 裁剪在其之上再加 **+5.10%**（C vs B），
两者可叠加。C 对 A 在 47/50 个样本上 TTFT 更优、50/50 吞吐更优；裁剪不损失吞吐
（−0.38%，在噪声内），因为它只动 prefill。

三个 arm 的答案一致性都是 50/50，token 一致性 49/50（唯一一次分歧发生在 greedy
平票处，top1−top2 logit 差为 0.0000，属数值 tie-break 而非正确性缺陷）。
decode 在 256 token 内保持平坦：四分位均值 `[6.33, 6.39, 6.33, 6.32]` ms。

早期 32 样本探针结果（`ppu_cudagraph_hybrid_ab_32samples.json`，0.75 / 54.384 ms /
46.743 → 52.387 ms / 156.864 tok/s，+3.67% / +235.59%）保留为历史证据。精度 0.75 与
0.76 的差异是样本量差异（32 vs 50），不是回归——两次 A/B 内部答案一致性都是全量一致。

开关：`--enable-hybrid-cudagraph --generation-runner greedy`
（等价环境变量 `OPTIZ_QWEN_CUDA_GRAPH_DECODE=1`）。prefill 裁剪默认开启，
用 `OPTIZ_QWEN_PREFILL_LAST_LOGIT_ONLY=0` 关闭。

组成：

- `Prefill / Decode` 分离调度
- CUDA Graph decode（在 `flash_attention_2` 下捕获、捕获后翻回 `sdpa`）
- `fla-core` Triton gated-delta-net prefill（+ `causal_conv1d` fast path，非加速项）
- prefill lm_head 只投影最后一个 position（`logits_to_keep=1`）
- `StaticCache`，默认 `max_cache_len=2048`

作为显存备选、非默认开启：`AWQ W4A16`、`FastV` 剪枝（`K=2`、`R≈40%~50%`）、
INT4 KV chain。

## 需要优先关注的风险

- 视觉 Token 剪枝可能伤害 OCR 和细粒度定位样本
- 量化若过于激进，跨模态层会先掉点
- 如果先做复杂投机解码，可能拖慢主线交付
- 若没有统一评测脚本，后续所有优化收益都难以证明

混合方案本身尚未覆盖的验证项（按优先级，不得当作已验证）：

1. `max_cache_len` > 2048 的性能与稳定性
2. 全量 MMBench 上精度是否成立（当前仅 50 样本，0.76）
3. 不同 prompt 长度的分桶策略
4. **prefill 的捕获/编译**——已确认 prefill 也是 dispatch-bound（`cpu_issue_fraction`
   0.988+），这是 TTFT 剩下的大杠杆，但未尝试
5. MTP（multi-token prediction）head 与混合方案的兼容性

已关闭：`causal_conv1d` 编译可用性（可编译、数值正确、对 TTFT 无实质影响，见上）。

因此本项目要求：

- 先建自动化评测，再做优化
- 先做低风险主线，再考虑增强项
- 所有结论以真实测试结果为准，不以论文数字替代本地验证
