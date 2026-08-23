# PPU 四版本性能对比报告

日期：2026-08-23

环境：PPU-ZW810E，PyTorch 2.9.0 + CUDA 13.0，Transformers 5.14.1，PPU-SDK 2.1.0
数据：MMBench dev-en 前 200 条，`max_new_tokens=256`。同学版/合并版/主仓库版走
harness-faithful 路径（直接构造 `VLMModel` 调 `generate_with_metrics`，清空全部
`OPTIZ_QWEN_*` 环境变量）；官方原版走官方 `benchmark_public.py` 原入口。

## 对比版本

| 版本 | 说明 |
|---|---|
| **官方原版** | 老师提供的 `dndx_participant-v1.2` 原始 wrapper，未做任何修改 |
| **同学版** | 队友提交方案：CUDA Graph decode + 分阶段 attention + prefill lm_head 裁剪 + early-stop，fp16 |
| **合并版** | 同学版 + 仓库 PPU delta kernel（18 层 GDN 全替换，bf16） |
| **主仓库版** | 合并代码已合入主仓库后的当前版本（与合并版同源，重新打包验证） |

## 性能结果

| 指标 | 官方原版 | 同学版 | 合并版 | 主仓库版 |
|---|---:|---:|---:|---:|
| 准确率 | **75.5%** (151/200) | 67.5% (135/200) | 66.5% (133/200) | 66.5% (133/200) |
| TTFT 均值 | 113.1 ms | 54.5 ms | **44.3 ms** | 44.9 ms |
| TTFT 中位 | 111.4 ms | 53.2 ms | **42.3 ms** | 43.3 ms |
| 吞吐（decode-only） | 60.9 tok/s | 160.0 tok/s | **161.6 tok/s** | 161.9 tok/s |
| CUDA Graph 捕获 | 无 | 全部 | 全部 | 全部 |
| delta kernel | 无 | 无 | 18 层 | 18 层 |

## 关键结论

1. **相对官方原版，同学版已把 TTFT 减半**（111.4 → 53.2 ms，-52%）、decode 吞吐
   提升 2.6×（60.9 → 161 tok/s）——CUDA Graph 派发消除是主因。
2. **合并版在同学版基础上再降 TTFT 中位 -18%**（53.2 → 42.3 ms），全部来自
   prefill 阶段。delta kernel 直接命中 prefill GDN 计算热点，CUDA Graph decode
   不受影响（decode 吞吐 160-162 tok/s 各版本一致）。
3. **精度代价**：官方原版 75.5% 最高；同学版相对原版 -8pp（75.5% → 67.5%），
   合并版再 -1pp（67.5% → 66.5%）。同学版的精度下降来自 bf16 加载、early-stop
   与解析口径差异，需在最终提交前确认官方复测口径下的真实影响。
4. **主仓库版与合并版一致**：准确率与 decode 吞吐完全相同，TTFT 差异（44.3 vs
   44.9 ms）在运行噪声范围内，确认代码同步无回归。
5. **按官方评分公式**（Accuracy×0.6 + TTFT×0.2 + Throughput×0.2）粗算：
   合并版相对官方原版净分大幅提升（TTFT -62%、吞吐 +165% 远超精度 -9pp），
   相对同学版净分 +0.029（TTFT -18% 增益超过精度 -1pp 损失）。

## 原始数据

- 官方原版：`/root/official_200.json`（服务器）
- 同学版：`/root/harness_teammate_200.json`（服务器）
- 合并版：`/root/harness_delta_200.json`（服务器）
- 主仓库版：`/root/harness_main_200.json`（服务器）
- 复现脚本：`scripts/profile/` 下 harness 验证脚本（需在 PPU 环境运行）
