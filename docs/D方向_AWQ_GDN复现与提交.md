# D 方向：AWQ / GDN CUDA 复现与提交

## 1. 这次提交包含什么

本分支以团队最新 `main` 的 `99f4c38` 为基线，只增加 D 方向的可回退实验入口。
没有把旧 AWQ 或 runtime 分支整体合并进来，也没有改变团队代码的默认推理行为。

| 组件 | 作用 | 默认状态 |
|---|---|---|
| AWQ W4A16 | 生成并加载 4-bit 权重、16-bit 激活的模型产物 | 关闭 |
| GDN CUDA fast path | 为 Qwen3.5 的 Gated DeltaNet 层启用 CUDA 算子 | 关闭 |
| CUDA 矩阵 runner | 在四种隔离模式间选择、检查资源并记录清单 | 无开关时只跑 baseline |

四种模式与命令行开关的对应关系：

| `--enable-awq` | `--enable-gdn-fastpath` | 实际模式 |
|---|---|---|
| 否 | 否 | 原始 FP16 baseline |
| 否 | 是 | GDN CUDA |
| 是 | 否 | AWQ |
| 是 | 是 | AWQ + GDN CUDA |

GDN 依赖必须放在独立 overlay 中。runner 会先确认：

1. baseline 环境看不到 GDN fast path；
2. 开启 GDN 后 CUDA fast path 确实可用；
3. 开启 AWQ 后，模型目录确实含有 compressed-tensors W4A16 元数据和权重。

资源不存在或检查失败时会直接报错，不会静默退回 baseline，也不会自动下载或现场
量化。

## 2. 对应实验结果

以下是 DNDX v1.1、4317 题全量、每组一轮的结果。速度的稳定性结论另以 10 题三轮
重复为准，详见 `reports/d_awq_gdn_results.md`。

| 方案 | 全量准确率 | TTFT | 吞吐 |
|---|---:|---:|---:|
| 原始 FP16 | 80.31% | 293.3 ms | 132.3 tok/s |
| GDN CUDA | 80.24% | 52.1 ms | 144.2 tok/s |
| AWQ | 82.26% | 294.6 ms | 124.5 tok/s |
| AWQ + GDN | 82.14% | 53.3 ms | 135.9 tok/s |

结果边界：

- 核心四组运行于提交 `e767deb`；它相对 `99f4c38` 只增加矩阵 runner 和测试。
- 当前整理版没有改模型计算逻辑，只把同一 runner 补成默认关闭的显式开关，并补齐
  AWQ/GDN 的准备与有效性检查。
- AWQ 模型目录比原始模型小 `34.92%`，但当前 Transformers 后端推理时会解压权重，
  稳定显存没有下降，不能称为 CUDA INT4 推理加速。
- 公开集准确率提升只是本次观测，不代表隐藏集一定提升。

## 3. 文件对应关系

| 文件 | 用途 |
|---|---|
| `scripts/run_v11_cuda_matrix.py` | 默认关闭的 AWQ/GDN 运行开关与四组 benchmark |
| `scripts/check_gdn_fastpath.py` | 检查 GDN CUDA 算子及 Transformers 路由 |
| `configs/requirements/runtime_gdn_cuda_py312.txt` | GDN overlay 的固定依赖 |
| `scripts/check_awq_cuda_readiness.py` | 检查隔离 AWQ 环境 |
| `scripts/prepare_awq_calibration.py` | 按固定 seed 生成 128 条校准样本清单 |
| `scripts/inventory_awq_modules.py` | 在 meta device 上确定精确量化目标 |
| `scripts/quantize_awq_w4a16.py` | dry-run 或显式生成 AWQ W4A16 权重 |
| `configs/experiments/d_awq_w4a16_cuda.json` | AWQ 算法、校准和环境合同 |
| `configs/requirements/d_awq_cuda_py312.txt` | AWQ 隔离环境依赖 |

约 3 GB 的 AWQ 权重、原始模型、数据集、虚拟环境、GDN 编译产物和 benchmark 原始
输出都不提交 Git。

## 4. 从零准备 AWQ 权重

以下示例使用 Linux、Python 3.12 和 CUDA 12.8。所有路径都可替换；脚本不依赖
作者服务器的固定目录。

### 4.1 创建隔离环境

```bash
python3.12 -m venv ~/.local/venvs/qwen-awq
PYTHON_BIN=~/.local/venvs/qwen-awq/bin/python

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$PYTHON_BIN" -m pip install \
  -r configs/requirements/d_awq_cuda_py312.txt
```

先检查环境和资源：

```bash
"$PYTHON_BIN" scripts/check_awq_cuda_readiness.py \
  --model-path /path/to/Qwen3.5-2B \
  --dataset-path /path/to/mmbench_dev_en.tsv \
  --output artifacts/awq/readiness.json
```

### 4.2 生成确定性的校准清单

固定参数为：128 条、seed `20260712`、排除 benchmark 前 10 题。公开 TSV 必须匹配
配置中的 SHA-256。

```bash
"$PYTHON_BIN" scripts/prepare_awq_calibration.py \
  --dataset-path /path/to/mmbench_dev_en.tsv \
  --output artifacts/awq/calibration_manifest.json
```

### 4.3 确定量化模块

此步骤只在 meta device 上构造模型结构，不加载 checkpoint 权重。视觉编码器、
multimodal projector、GDN/linear-attention、embedding、norm 和 `lm_head` 都被排除；
只量化语言 MLP 与 full-attention 中的 Linear。

```bash
"$PYTHON_BIN" scripts/inventory_awq_modules.py \
  --model-path /path/to/Qwen3.5-2B \
  --output artifacts/awq/module_inventory.json
```

### 4.4 先 dry-run，再显式量化

不写 `--execute` 时只校验合同和 recipe，不生成权重：

```bash
"$PYTHON_BIN" scripts/quantize_awq_w4a16.py \
  --model-path /path/to/Qwen3.5-2B \
  --dataset-path /path/to/mmbench_dev_en.tsv \
  --calibration-manifest artifacts/awq/calibration_manifest.json \
  --module-inventory artifacts/awq/module_inventory.json \
  --report artifacts/awq/dry_run.json
```

确认 dry-run 后再执行；输出目录必须尚不存在：

```bash
"$PYTHON_BIN" scripts/quantize_awq_w4a16.py \
  --model-path /path/to/Qwen3.5-2B \
  --dataset-path /path/to/mmbench_dev_en.tsv \
  --calibration-manifest artifacts/awq/calibration_manifest.json \
  --module-inventory artifacts/awq/module_inventory.json \
  --output-dir artifacts/awq/qwen35_2b_w4a16 \
  --report artifacts/awq/quantization_report.json \
  --execute
```

量化脚本本身不把生成前 baseline 当成生成权重的必要条件；性能结论必须在生成后用
下面的 DNDX v1.1 runner 与同环境 baseline 对照。

## 5. 准备隔离的 GDN CUDA overlay

不要把 `fla-core` 和 `causal-conv1d` 安装进 baseline/AWQ 虚拟环境，否则
Transformers 可能在没有开关时自动启用 fast path，污染基线。

```bash
PYTHON_BIN=~/.local/venvs/qwen-awq/bin/python
GDN_OVERLAY=~/.local/overlays/qwen-gdn-fastpath

mkdir -p "$GDN_OVERLAY"
"$PYTHON_BIN" -m pip install \
  --target "$GDN_OVERLAY" \
  --no-deps \
  -r configs/requirements/runtime_gdn_cuda_py312.txt

PYTHONPATH="$GDN_OVERLAY:$PWD/src" \
  "$PYTHON_BIN" scripts/check_gdn_fastpath.py --require-cuda
```

`--no-deps` 是为了不把 overlay 中的包和基础环境混装；基础环境应已安装与实验匹配的
Torch、Transformers 和 Triton。

## 6. 运行四种模式

先定义公共参数：

```bash
PYTHON_BIN=~/.local/venvs/qwen-awq/bin/python
MODEL=/path/to/Qwen3.5-2B
AWQ_MODEL=/path/to/qwen35_2b_w4a16
DATASET=/path/to/mmbench_dev_en.tsv
GDN_OVERLAY=~/.local/overlays/qwen-gdn-fastpath
```

### 原始 FP16（两个开关都不写）

```bash
"$PYTHON_BIN" scripts/run_v11_cuda_matrix.py \
  --dataset-path "$DATASET" \
  --baseline-model-path "$MODEL" \
  --output-root benchmarks/output/baseline \
  --python "$PYTHON_BIN"
```

### GDN CUDA

```bash
"$PYTHON_BIN" scripts/run_v11_cuda_matrix.py \
  --dataset-path "$DATASET" \
  --baseline-model-path "$MODEL" \
  --gdn-overlay "$GDN_OVERLAY" \
  --output-root benchmarks/output/gdn \
  --python "$PYTHON_BIN" \
  --enable-gdn-fastpath
```

### AWQ

```bash
"$PYTHON_BIN" scripts/run_v11_cuda_matrix.py \
  --dataset-path "$DATASET" \
  --baseline-model-path "$MODEL" \
  --awq-model-path "$AWQ_MODEL" \
  --output-root benchmarks/output/awq \
  --python "$PYTHON_BIN" \
  --enable-awq
```

### AWQ + GDN

```bash
"$PYTHON_BIN" scripts/run_v11_cuda_matrix.py \
  --dataset-path "$DATASET" \
  --baseline-model-path "$MODEL" \
  --awq-model-path "$AWQ_MODEL" \
  --gdn-overlay "$GDN_OVERLAY" \
  --output-root benchmarks/output/awq_gdn \
  --python "$PYTHON_BIN" \
  --enable-awq \
  --enable-gdn-fastpath
```

默认协议是前 10 题、3 次、warmup 2、max new tokens 256。正式全量时增加
`--num-samples 0 --repeats 1`。高级矩阵模式仍可使用 `--cases ...`，但不能与两个
默认关闭的开关混用。

每个输出目录都会产生：

- `运行清单.json`：Git、数据 SHA-256、模型身份、软件、GPU、请求开关和实际检查；
- 每个 case 的 `runN.json` 与 `runN.log`；
- `汇总.json`：均值、标准差和相对 baseline 变化。

## 7. Git 提交边界

建议 PR 只包含本页第 3 节列出的源码、配置、测试和报告。不要提交：

- `resources/**/raw/` 下的模型或数据；
- `artifacts/` 下生成的 AWQ 权重；
- `benchmarks/output/` 下大体积运行结果；
- 虚拟环境、overlay、CUDA 编译缓存；
- 任何账号、密码、私钥或服务器专用认证文件。
