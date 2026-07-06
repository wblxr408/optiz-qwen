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

## 技术路线初版

- 视觉压缩：`FastV` 风格层间视觉 Token 剪枝
- 权重量化：`AWQ W4A16`
- 运行时调度：`Prefill / Decode` 分离
- KV 管理：分页 KV Cache
- Kernel 优化：`Dequant + GEMM/GEMV` 融合，优先 Attention 与 FFN 主路径
- PPU 适配：算子兼容性扫描、长尾算子补齐、packed weight 布局优化

增强分支而非首发主线：

- `SparseVLM` 风格文本引导稀疏化
- `KIVI` 风格 KV 低比特量化
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

## 当前推荐提交版本

- `AWQ W4A16`
- `FastV` 风格剪枝，默认 `K=2`、`R≈40%~50%`
- `Prefill / Decode` 分离调度
- `Dequant + GEMM/GEMV` 融合
- `PPU` packed weight 与长尾算子补齐

## 需要优先关注的风险

- 视觉 Token 剪枝可能伤害 OCR 和细粒度定位样本
- 量化若过于激进，跨模态层会先掉点
- 如果先做复杂投机解码，可能拖慢主线交付
- 若没有统一评测脚本，后续所有优化收益都难以证明

因此本项目要求：

- 先建自动化评测，再做优化
- 先做低风险主线，再考虑增强项
- 所有结论以真实测试结果为准，不以论文数字替代本地验证
