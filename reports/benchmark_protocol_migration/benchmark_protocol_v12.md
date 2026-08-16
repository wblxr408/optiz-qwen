# DNDX 官方 v1.2 解析规则更新记录

日期：2026-08-15

主办方发布 `dndx_participant-v1.2.zip`。与 v1.1 逐文件比较后，唯一实质性
变化是 `benchmark_public.py` 的答案解析正则：新增对 Markdown 强调/行内代码
标记的容忍，评分流程、数据集、wrapper 与 README 均未变化。

## v1.2 正则变化

新增两个公共片段：

```python
ANSWER_MARK = r"(?:\*{1,3}|_{1,3}|`{1,3})*"
ANSWER_CHOICE = rf"{ANSWER_MARK}\s*[\(\[（【]?\s*([ABCD])\s*[\)\]）】]?\s*{ANSWER_MARK}"
```

三条 `ANSWER_PATTERNS` 均改为复用 `ANSWER_CHOICE`，使官方解析器现在能从
`**B**`、`*答案：* C *`、`正确答案是 **D**。`、`` `Answer: A` `` 等带
加粗/斜体/行内代码标记的输出中提取选项。v1.1 对这些格式返回 `None`。

## 对仓库的影响

- `src/optiz_qwen/evaluation/answer_parsing.py`：`ANSWER_PATTERNS` 已更新为
  与官方 v1.2 逐字节一致；`parse_choice_answer` 的来源标记改为
  `official_v1.2_pattern`。
- `src/optiz_qwen/evaluation/dndx_public_benchmark.py`：输出
  `benchmark_version` 改为 `dndx_public_self_test_v1.2`。
- `scripts/benchmark_kv_paired.py` 与 `scripts/benchmark_tome_paired.py`：
  通过 `parse_choice_answer` 间接获得 v1.2 解析行为，无需修改。
- `scripts/run_v11_cuda_matrix.py` 保持 v1.1 断言不变，仅用于重放历史 v1.1
  比较，不再与新基准输出混用。

## 数据一致性注意

v1.2 解析器只会把 v1.1 判为“无法提取”的部分输出（带 Markdown 标记的答案）
转为可得分，因此新旧解析器对同一批原始输出不会产生冲突，只会让解析结果
变多。旧结果文件如需对齐新口径，可用 v1.2 解析器对其中记录的原始输出重放
评分，而不必重跑模型。
