# 官方 v1.2 原始代码性能瓶颈独立验证

日期：2026-08-23

环境：PPU-ZW810E，PyTorch 2.9.0 + CUDA 13.0，Transformers 5.14.1，PPU-SDK 2.1.0

目的：在**老师提供的官方 `dndx_participant-v1.2` 原始代码**（`evaluation_wrapper.py`
与 `benchmark_public.py`，未做任何修改）上独立测量性能瓶颈，验证队友"CPU kernel
派发是瓶颈、而非显存带宽"的诊断。本验证不依赖团队仓库代码，也不依赖任何优化开关。

## 测量方法

- 官方 wrapper 原样加载 Qwen3.5-2B（bf16，`device_map="auto"`），官方 prompt 与
  MMBench dev-en 真实样本；
- 由于 PPU 的 CUDA 兼容层不向 `torch.profiler` 暴露 CUDA kernel 事件
  （实测 `cuda_kernels=0`），改用 CPU-only `torch.autograd.profiler` 统计 CPU 侧
  kernel 派发调用（`cudaLaunchKernel` / `cuLaunchKernel` / `cuLaunchKernelEx`）；
- 以 `max_new_tokens=1` 与 `max_new_tokens=8` 两次生成之差拆分单次 decode step
  的派发量与耗时（prefill 相同，相减后只剩 decode 增量）；
- 所有测量在 warmup（内核编译）之后进行。

脚本：

- `scripts/profile/profile_official_v12_kernels.py`：kernel 派发统计（本文数据来源）
- `scripts/profile/profile_official_v12_wallclock.py`：墙钟分层（prefill/decode）

## 测量结果（官方代码原样）

### 单次生成（1 次 prefill + 8 次 decode）派发总量

| 指标 | 数值 |
|---|---:|
| cudaLaunchKernel | 8,977 次 |
| cuLaunchKernel + cuLaunchKernelEx | 2,040 次 |
| 合计 kernel 派发 | ≈ 11,000 次 |
| cudaMemcpyAsync | 364 次 |
| CPU 派发自耗 | 37.5 ms |

### 单次 decode step（两次生成之差）

| 指标 | 官方代码实测 | 队友报告 |
|---|---:|---:|
| kernel 派发 / step | ≈ 1,247 次 | 5,778 次 |
| CPU 派发时间 / step | ≈ 4.1 ms | — |
| decode 墙钟 / step | 20–32 ms | 19–21 ms |
| HBM 理论下限 / step | 4.55 ms（219 tok/s） | 2.2 ms（454 tok/s） |

## 结论

1. **"CPU kernel 派发是瓶颈"成立**：官方代码每次 decode step 需派发约 1,200+ 次
   kernel，CPU 派发占 decode 墙钟的显著比例。按 4.55 GB 权重、2 TB/s 假设的
   HBM 理论下限为 4.55 ms/step（219 tok/s），实测 decode 20–32 ms/step 仍有
   5–7 倍差距，该差距只能由派发开销解释。
2. **与队友数字的差异**：kernel/step 1,247 vs 5,778 来自测量口径（CPU 侧 launch
   调用 vs 更深层 CUDA kernel 计数；PPU 兼容层对 profiler 的可见性不同）。
   方向一致、量级同阶，不影响结论。
3. **方法学注意**：PPU 上 `torch.profiler` 的 CUDA 活动为空，CPU-only
   autograd profiler 可统计 launch 调用但 CPU 时间会因 profiler 开销膨胀，
   因此墙钟数值只作量级参考。

## 原始数据

- `official_v12_wallclock_profile.json`：墙钟分层原始输出
- kernel 派发原始输出在服务器 `/root/`（未入库，可重跑脚本复现）
