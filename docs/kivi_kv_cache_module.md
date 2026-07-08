
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

---

# 当前状态（Current State）

## 已完成（Implemented）

* 检查上游源码，并记录 Commit 与 License 信息。
* 支持直接加载上游：

  * `LlamaForCausalLM_KIVI`
  * `MistralForCausalLM_KIVI`
* 支持直接加载上游 `quant.new_pack` 中的 pack / unpack 实现。
* 为 **Qwen3.5 VLM** 编写 `DynamicCache` Adapter，仅替换 **full_attention** 层为 KIVI Packed Cache。
* Prefill 阶段仍向原生 Attention 返回完整 K/V Tensor，同时内部保留量化后的 Packed Cache，这比立即解包（unpack）更符合 KIVI 论文中的 Prefill 行为。
* Decode 阶段采用**逐 Token 更新**，避免每一步都重新量化整个历史 KV Cache。
* 通过环境变量开启 DNDX Wrapper：

```
OPTIZ_QWEN_KIVI_KV_CACHE=1
```

* DNDX Wrapper 能够传播生成线程中的异常，而不会因为 Streamer 导致程序挂起。
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
* 默认 DNDX Benchmark Wrapper **不会启用 KIVI**，必须显式设置环境变量。
* 尚未声明兼容 PPU（Processing Processing Unit）。
* 尚未获得任何优化后的 Benchmark 数据。
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

但公开评测（Public Validation）仍失败，错误为：

```
missing_choice_answer
```

因此**准确率保持（Accuracy Retention）尚未验证**。

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
