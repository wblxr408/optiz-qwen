# DNDX 官方 v1.1 评测协议迁移报告

日期：2026-07-22

## 摘要

主办方发布 `dndx_participant-v1.1.zip` 后，本项目重新核对官方基线并迁移本地
评测协议。新版压缩包只有 `benchmark_public.py` 发生实质变化：生成上限从
64 Token 提高到 256 Token，答案解析从单一正则扩展为三组官方模式。

本次迁移删除了本地评分器中的选项文本反推逻辑，并将 public benchmark 与
ToMe 配对脚本的默认上限统一为 256。此后所有正式准确率、吞吐率和端到端耗时
结论都应使用 v1.1；旧结果只保留为历史探索证据。

## 官方文件核验

对初版目录与 v1.1 压缩包逐文件比较后得到：

- `evaluation_wrapper.py` 未变化；
- `requirements.txt` 未变化；
- EN/CN 两份 MMBench TSV 的 SHA-1 完全一致；
- `README.md` 未变化；
- 只有 `benchmark_public.py` 修改了答案正则和 `max_new_tokens`。

因此不需要重新下载模型、数据集或依赖，也不需要改变 wrapper 接口。

## 协议变化

### 生成上限

| 项目 | 旧协议 | 官方 v1.1 |
|---|---:|---:|
| `max_new_tokens` | 64 | 256 |

真实 CN5 冒烟测试中，样本 241 生成了 65 Token，并被官方规则正确计分；旧协议
会在它完成前截断。最终 CN50 复测进一步表明：

| 模式 | 旧协议命中 64 上限 | v1.1 中超过 64 | v1.1 命中 256 上限 |
|---|---:|---:|---:|
| Baseline | 9/50 | 8/50 | 0/50 |
| ToMe L16R32 | 5/50 | 5/50 | 1/50 |

英文两组没有受到上限影响：前 50 条最大生成长度分别为 54 和 47，后 50 条两组
最大值均为 24。

### 答案解析

旧本地解析器包含 `exact_choice_text`：当输出没有显式 A/B/C/D 时，根据完整选项
文本反推答案。该逻辑不属于官方评分规则，现已删除。

将官方 v1.1 正则重放到旧版最终 CN50 baseline 原始输出上：

- 本地增强解析：43/50；
- 官方 v1.1 解析：41/50；
- 5 条解析结果不同：241、258、308、313、407；
- 这 5 条在官方规则下均无法直接提取选项。

`evaluation_wrapper.py` 内已有的 logits 选项回退仍然保留。它在模型原始文本无法
通过官方正则时额外执行一次模型 forward，并输出明确的 `Answer: X`。这属于
wrapper 内部推理策略，而不是改写 benchmark 评分器。为避免隐藏性能代价，新版
报告必须披露它的触发次数。

## 代码迁移

- `answer_parsing.py`：逐字同步官方三组正则，移除选项文本反推；
- `dndx_public_benchmark.py`：默认上限改为 256，输出协议版本和生成配置；
- `benchmark_tome_paired.py`：默认上限改为 256，结果标记为
  `tome_paired_v2_dndx_v1.1`；
- `test_dndx_entrypoints.py`：覆盖官方新增表达，并证明评分器不会越权推断答案。

验证结果：`62 passed, 1 skipped`。dummy 冒烟和真实 CN5 均通过 public
validation。

## 新版正式基线

最终候选的 v1.1 配对复测覆盖 EN100 + CN50。Baseline 汇总如下：

| 数据 | 正确率 | 平均 TTFT | Decode 吞吐 | 生成 Token | Logits 回退 |
|---|---:|---:|---:|---:|---:|
| EN100 | 78/100 | 959.44 ms | 15.118 tok/s | 988 | 0 |
| CN50 | 43/50 | 921.38 ms | 10.075 tok/s | 2374 | 5 |
| 合计 | 121/150 | 946.75 ms | 13.437 tok/s | 3362 | 5 |

结果文件：

- `tome_prop_l16_r32_en50_v11_256tok_paired_baseline.json`
- `tome_prop_l16_r32_en50_offset50_v11_256tok_paired_baseline.json`
- `tome_prop_l16_r32_cn50_v11_256tok_paired_baseline.json`

这些 JSON 位于 `benchmarks/output/`，该目录按项目约定不提交 Git；可复现结论和
图像记录在 `reports/A/tome_complete_evaluation.md`。

## 如何解释旧结果

旧报告并非全部失效：视觉编码器剖析、ToMe 算子 microbenchmark、视觉 Token 数量
和参数筛选仍能解释优化机制。受影响的是以下指标：

- 准确率：旧本地解析器与官方规则不同；
- 完整耗时：中文回答曾被 64 Token 截断；
- Decode 吞吐：生成长度和回退次数发生变化；
- 跨日期绝对 TTFT：两轮运行相隔 9 天，系统和热状态不同。

因此旧结果只能用于历史机制分析，不能与 v1.1 的绝对时间直接计算“协议升级带来
多少加速”。今后的优化比较必须在同一轮 v1.1 paired run 中完成。
