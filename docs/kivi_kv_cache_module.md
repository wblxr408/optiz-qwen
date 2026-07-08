
# KIVI KV Cache 模块

## 模块范围（Scope）

这是一个**独立的 KV Cache 量化增强模块**，基于上游 `jy-yuan/KIVI` 项目实现：

`https://github.com/jy-yuan/KIVI`

该模块属于 `docs/vlm_optimization_full_stack_architecture.svg` 中的**模型压缩层（Model Compression Layer）**。

默认基线（baseline）流程不会启用该模块。

---

# 决策记录（Decision Log）

采用**外部上游仓库**：

```
artifacts/third_party/KIVI
```

并通过本地 Adapter：

```
src/optiz_qwen/compression/kivi_external.py
```

进行集成。

## 2026-07-08 补充决策

### DNDX 中启用 KIVI 的方式

候选方案：

* 方案 A：继续只依赖 `OPTIZ_QWEN_KIVI_KV_CACHE=1`。
* 方案 B：在 `benchmark_public.py` 增加显式 CLI 开关，同时保留环境变量兼容。

评价维度：

* 性能：两者无直接差异。
* 成本：方案 B 只增加轻量参数解析和临时环境变量上下文。
* 维护性：方案 B 更容易复现实验命令，减少“忘记设置环境变量”的风险。
* 兼容性：方案 B 不改变 baseline 默认行为，仍符合“默认不启用增强分支”的仓库约束。

最终选择：

* 采用方案 B，新增：

```powershell
python benchmark_public.py --enable-kivi-kv-cache
```

拒绝方案 A 的原因：

* 环境变量仍可用，但单独依赖它不利于保存可复现 Benchmark 命令。

### Public Validation 的答案格式修复

候选方案：

* 方案 A：只扩大正则表达式，继续解析自由生成文本。
* 方案 B：先解析自由生成文本；若无法得到 A/B/C/D，再用同一个 VLM 对候选项做 logits 约束选择，并在输出中规范化为 `Answer: X`。

评价维度：

* 准确率真实性：方案 B 的兜底答案来自模型 logits，而不是硬编码或随机填充。
* 兼容性：方案 B 仍保留原始自由生成文本到 `meta.raw_text`，便于审计。
* 成本：方案 B 在失败样本上额外执行一次前向计算，延迟会增加。
* 鲁棒性：方案 B 能解决模型没有显式写出字母导致的 `missing_choice_answer`。

最终选择：

* 采用方案 B，并允许通过以下环境变量关闭：

```powershell
$env:OPTIZ_QWEN_CHOICE_FALLBACK='0'
```

拒绝方案 A 的原因：

* 只靠正则无法处理模型用自然语言复述答案但不输出选项字母的情况。

### qB MatMul Kernel 接入

候选方案：

* 方案 A：立即将 Qwen3.5 VLM Attention 路径替换为上游 qB MatMul Kernel。
* 方案 B：当前阶段继续使用上游 `quant.new_pack` pack/unpack，保留普通 K/V Tensor 供 Transformers 原生 Attention 使用。

评价维度：

* 正确性：方案 A 需要完整替换 Attention 计算路径，风险高。
* 兼容性：上游 KIVI 当前 Commit 未提供 Qwen3.5 VLM 原生实现。
* 维护成本：方案 A 需要额外 CUDA extension、形状约束、Attention mask 与 mixed attention 层适配。
* 比赛优先级：在 baseline/evaluation 尚未完全稳定前，方案 B 更符合先评测后 kernel 深改的顺序。

最终选择：

* 暂用方案 B。qB MatMul Kernel 仍为未接入项，不能声明 kernel 优化收益。
* 新增 `inspect_qb_matmul_kernel()` 作为能力探测；它只能证明上游 qB Kernel 是否可导入，不能证明已替换 Qwen3.5 VLM Attention。

拒绝方案 A 的原因：

* 缺少 Qwen3.5 VLM 官方 KIVI Attention 路径和 PPU/目标环境验证，不满足当前 Definition of Done。

---

# 当前状态（Current State）

## 已完成（Implemented）

* 检查上游源码，并记录 Commit 与 License 信息。
* 支持直接加载上游：

  * `LlamaForCausalLM_KIVI`
  * `MistralForCausalLM_KIVI`
* 支持直接加载上游 `quant.new_pack` 中的 pack / unpack 实现。
* 支持检查上游 `quant.matmul` qB MatMul Kernel 的可导入状态和缺失依赖。
* 为 **Qwen3.5 VLM** 编写 `DynamicCache` Adapter，仅替换 **full_attention** 层为 KIVI Packed Cache。
* Prefill 阶段仍向原生 Attention 返回完整 K/V Tensor，同时内部保留量化后的 Packed Cache，这比立即解包（unpack）更符合 KIVI 论文中的 Prefill 行为。
* Decode 阶段采用**逐 Token 更新**，避免每一步都重新量化整个历史 KV Cache。
* 通过环境变量开启 DNDX Wrapper：

```
OPTIZ_QWEN_KIVI_KV_CACHE=1
```

* DNDX Wrapper 能够传播生成线程中的异常，而不会因为 Streamer 导致程序挂起。
* DNDX Public Benchmark 支持通过 CLI 显式启用 KIVI：

```
--enable-kivi-kv-cache
```

* Public Validation 增加答案规范化路径：优先解析生成文本；若缺少 A/B/C/D，则使用模型 logits 对可用选项做约束选择，并记录 `answer_source` 与 `meta.raw_text`。
* 提供配置文件：

```
configs/models/kivi_kv_cache.json
```

* 提供可复现的上游仓库下载脚本：

```
scripts/prepare_kivi_upstream.ps1
```

* 完成以下单元测试：

  * 上游源码检查
  * 配置加载
  * Qwen 不支持情况的检测

---

## 尚未完成（Not completed）

* 当前 Commit 的上游 KIVI **尚未直接支持 Qwen3.5 VLM**。

  因此本项目通过本地 Cache 类完成适配，同时继续使用官方 KIVI 的 pack/unpack 实现。
* 默认 DNDX Benchmark Wrapper **不会启用 KIVI**。这是刻意保留的 baseline 约束；现在可通过 CLI 或环境变量显式启用。
* 尚未声明兼容 PPU（Processing Processing Unit）。当前只新增了 `inspect_ppu_compatibility()` 状态报告，结果仍为 `unverified`。
* 尚未获得完整、公开验证通过的优化 Benchmark 对比数据。
* 当前 Adapter：

  * 内部以 KIVI Packed 格式保存 KV Cache；
  * 推理时恢复为普通 K/V Tensor，供 Transformers 原生 Attention 使用。

  **尚未替换为官方 qB MatMul Kernel。**
* 安装

```
triton-windows==3.7.1.post27
```

后，可成功导入：

```
quant.new_pack
```

* 单样本 CUDA Smoke Test 已能够在

```
OPTIZ_QWEN_KIVI_KV_CACHE=1
```

下运行完成。

2026-07-08 复测后，原样本 `241` 的公开验证格式错误已修复。原因不是模型没有给出答案，而是旧解析器无法识别模型复述的完整选项文本。

当前仍不能声明完整**准确率保持（Accuracy Retention）**，因为只完成了 10 样本小批量 smoke 对比，尚未跑完整 public dev 集或官方评测集。

---

## 当前模型边界（Known Model Boundary）

本地 Qwen3.5-2B 配置采用混合 Attention 架构：

* `linear_attention`
* `full_attention`

Adapter 保留：

* 原生 Linear Attention 的 Conv/Recurrent Cache

仅替换以下 Full Attention 层：

```
(3, 7, 11, 15, 19, 23)
```

为 KIVI Cache。

---

# 本地验证（Local Validation）

2026-07-08，在 `optiz-qwen` Conda 环境中执行：

```powershell
python -m pytest

C:\Users\wblxr\anaconda3\envs\optiz-qwen\python.exe -m pytest

$env:OPTIZ_QWEN_KIVI_KV_CACHE='1'

C:\Users\wblxr\anaconda3\envs\optiz-qwen\python.exe benchmark_public.py `
    --backend transformers `
    --device cuda `
    --num-samples 1 `
    --warmup-samples 0 `
    --max-new-tokens 64 `
    --dataset-path resources\eval_dataset\raw\mmbench_public\mmbench_dev_cn.tsv `
    --output benchmarks\output\kivi_cuda_cn_1x64_smoke.json
```

---

## 运行结果（Observed Results）

* 单元测试：

  基础 Python 环境与 `optiz-qwen` Conda 环境均为：

```
15 passed
```

* Smoke Test 输出：

```
benchmarks/output/kivi_cuda_cn_1x64_smoke.json
```

* KIVI Smoke 状态：

  * Generation 成功完成；
  * `meta.kivi_kv_cache.enabled = true`；
  * 启用了以下 Full Attention 层：

```
(3, 7, 11, 15, 19, 23)
```

---

仍存在的问题：

```
public_validation_passed = false
```

原因：

样本 **241** 的生成结果无法解析为评测要求的：

```
A / B / C / D
```

因此公开验证尚未通过。

---

上游 Warning：

```
quant/new_pack.py
```

在当前 PyTorch 中会产生：

> 多维非 Tuple 索引（non-tuple multidimensional indexing）已弃用（Deprecation Warning）。

目前不会影响运行，但在未来升级 PyTorch 前应跟踪处理。

## 2026-07-08 小批量对比复测

运行环境：

```powershell
C:\Users\wblxr\anaconda3\envs\optiz-qwen\python.exe
```

Baseline 10 样本：

```powershell
C:\Users\wblxr\anaconda3\envs\optiz-qwen\python.exe benchmark_public.py `
    --backend transformers `
    --device cuda `
    --num-samples 10 `
    --warmup-samples 0 `
    --max-new-tokens 64 `
    --dataset-path resources\eval_dataset\raw\mmbench_public\mmbench_dev_cn.tsv `
    --output benchmarks\output\baseline_cuda_cn_10x64_validation_fix.json
```

KIVI 10 样本：

```powershell
C:\Users\wblxr\anaconda3\envs\optiz-qwen\python.exe benchmark_public.py `
    --backend transformers `
    --device cuda `
    --num-samples 10 `
    --warmup-samples 0 `
    --max-new-tokens 64 `
    --dataset-path resources\eval_dataset\raw\mmbench_public\mmbench_dev_cn.tsv `
    --enable-kivi-kv-cache `
    --output benchmarks\output\kivi_cuda_cn_10x64_validation_fix.json
```

对比：

```powershell
python scripts\compare_benchmarks.py `
    --baseline cn=benchmarks\output\baseline_cuda_cn_10x64_validation_fix.json `
    --candidate cn=benchmarks\output\kivi_cuda_cn_10x64_validation_fix.json `
    --plot benchmarks\output\compare_baseline_vs_kivi_cn_10x64_validation_fix.png
```

结果：

| 指标 | Baseline | KIVI | 结论 |
| --- | ---: | ---: | --- |
| 样本数 | 10 | 10 | 小批量 smoke，不代表 4029 条完整 public dev |
| public validation | passed | passed | `missing_choice_answer` 已修复 |
| accuracy | 1.000 | 1.000 | 10 样本保持 |
| avg TTFT ms | 914.8 | 1686.4 | KIVI 更慢 |
| throughput tokens/s | 8.75 | 6.11 | KIVI 更慢 |

因此当前只能声明：

* 10 样本 CUDA public validation 已通过；
* 10 样本准确率未回退；
* 尚无优化性能收益，不能声明 KIVI 已带来 TTFT 或吞吐提升；
* 完整 Accuracy Retention 仍未完成。

## qB MatMul Kernel 当前探测结果

在 `optiz-qwen` Conda 环境中执行：

```powershell
C:\Users\wblxr\anaconda3\envs\optiz-qwen\python.exe -c "import sys; sys.path.insert(0, r'D:\optiz-qwen\src'); from optiz_qwen.compression import inspect_qb_matmul_kernel; print(inspect_qb_matmul_kernel())"
```

当前上游 `quant.matmul` 导入失败：

```text
ModuleNotFoundError: No module named 'kivi_gemv'
```

含义：

* `quant.new_pack` 可用；
* 上游 qB MatMul CUDA extension 尚未装好或不可导入；
* Qwen3.5 VLM Adapter 尚未替换为官方 qB MatMul Kernel；
* 当前不能声明 kernel 层优化完成。

---

# 环境搭建（Setup）

PowerShell：

```powershell
.\scripts\prepare_kivi_upstream.ps1
```

随后安装上游项目：

```powershell
pip install -e .\artifacts\third_party\KIVI
```

安装 Triton：

```powershell
pip install .\artifacts\wheels\triton_windows-3.7.1.post27-cp311-cp311-win_amd64.whl
```

安装量化扩展：

```powershell
Push-Location .\artifacts\third_party\KIVI\quant

pip install -e .

Pop-Location
```

---

## 关于 CUDA Extension

`quant` 目录中的 CUDA 扩展**仅用于上游 qB MatMul Kernel**。

当前 Qwen3.5 的 KV Cache Adapter 仅依赖：

* `quant.new_pack`
* `triton-windows`

因此，在**未编译 CUDA Extension** 的情况下，也可以完成当前版本的功能测试。
