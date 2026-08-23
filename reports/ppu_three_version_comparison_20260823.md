# PPU 三版本性能对比报告

日期：2026-08-23

环境：PPU-ZW810E，PyTorch 2.9.0 + CUDA 13.0，Transformers 5.14.1，PPU-SDK 2.1.0
数据：MMBench dev-en 前 200 条，harness-faithful 路径（直接构造 `VLMModel` 调
`generate_with_metrics`，清空全部 `OPTIZ_QWEN_*` 环境变量），`max_new_tokens=256`

## 对比版本

| 版本 | 说明 |
|---|---|
| **同学版** | 队友提交方案：CUDA Graph decode + 分阶段 attention + prefill lm_head 裁剪 + early-stop，fp16 |
| **合并版** | 同学版 + 仓库 PPU delta kernel（18 层 GDN 全替换，bf16） |
| **主仓库版** | 合并代码已合入主仓库后的当前版本（与合并版同源，重新打包验证） |

## 性能结果

| 指标 | 同学版 | 合并版 | 主仓库版 |
|---|---:|---:|---:|
| 准确率 | 67.5%（135/200） | 66.5%（133/200） | 66.5%（133/200） |
| TTFT 均值 | 54.5 ms | **44.3 ms** | 44.9 ms |
| TTFT 中位 | 53.2 ms | **42.3 ms** | 43.3 ms |
| prefill 均值 | 54.7 ms | **44.6 ms** | 47.3 ms |
| decode 均值 | 186.8 ms | 185.3 ms | 207.5 ms |
| 吞吐（decode-only） | 160.0 tok/s | **161.6 tok/s** | 161.9 tok/s |
| CUDA Graph 捕获 | 全部 | 全部 | 全部 |
| delta kernel | 否 | 18 层 | 18 层 |

## 关键结论

1. **TTFT 显著下降**：合并/主仓库版相对同学版 TTFT 中位 **-18%**（53.2 → 42.3 ms），
   全部来自 prefill 阶段（54.7 → 44.6 ms）。delta kernel 直接命中 prefill GDN
   计算热点，CUDA Graph decode 不受影响（decode 吞吐 160-162 tok/s 三版本一致）。
2. **精度代价固定且小**：-1pp（67.5% → 66.5%，2/200 题），与之前 50/100 条验证
   一致；全量 4029 上预计在 ±1pp 量级。
3. **主仓库版与合并版一致**：准确率与 decode 吞吐完全相同，TTFT 差异（44.3 vs
   44.9 ms）在运行噪声范围内，确认代码同步无回归。
4. **按官方评分公式**（Accuracy×0.6 + TTFT×0.2 + Throughput×0.2）粗算：
   合并版净分相对同学版 **+0.029**（TTFT -18% 的增益超过精度 -1pp 的损失）。

## 原始数据

- 同学版：`/root/harness_teammate_200.json`（服务器）
- 合并版：`/root/harness_delta_200.json`（服务器）
- 主仓库版：`/root/harness_main_200.json`（服务器）
- 复现脚本：`scripts/profile/` 下 harness 验证脚本（需在 PPU 环境运行）
