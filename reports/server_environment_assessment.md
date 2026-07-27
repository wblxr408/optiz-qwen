# 学校公共 PPU 服务器环境简报

日期：2026-07-27

## 1. 检查范围

本次仅通过只读命令检查系统、硬件、SDK 和 Python 框架版本，没有上传团队仓库、
优化代码、实验配置或凭据，也没有修改服务器文件。

该服务器由所有参赛同学共享，并且使用具有高权限的公共账号。因此服务器上的目录、
文件权限和虚拟环境均不能视为团队隔离边界。

## 2. 当前环境

| 项目 | 检查结果 |
|---|---|
| 操作系统 | Ubuntu 24.04.4 LTS，x86_64 |
| 运行形态 | Kata Containers / overlay 容器环境 |
| CPU | 44 个 Intel Xeon vCPU |
| 系统内存 | 约 440 GiB |
| PPU | 4 张 PPU-ZW810E |
| 单卡显存 | 97,920 MiB，约 95.62 GiB |
| PPU SDK | 2.1.1 |
| PPU 驱动 | 1.3.2 |
| Python | 3.12.3 |
| PyTorch | 2.6.0 |
| PyTorch CUDA 接口版本 | 12.6 |
| Transformers | 5.14.1 |
| Triton | 3.7.1 |
| 持久工作盘 | 约 30 GiB，检查时剩余约 28 GiB |

检查时四张 PPU 均为空闲状态。服务器已经预装 PPU SDK、`ppu-smi`、PPU
ModelZoo、PPU 版 Triton 开发目录，以及官方参赛包和 Qwen3.5-2B 权重。

## 3. 兼容性判断

PPU 通过 PyTorch 的 CUDA 接口暴露：

- `torch.cuda.is_available()` 返回 `True`；
- `torch.cuda.device_count()` 返回 `4`；
- PyTorch 可读取四张 `PPU-ZW810E` 的设备属性。

这说明普通 PyTorch CUDA 接口可能无需大量改动即可运行，但不能据此认定下列组件已经
兼容：

- NVIDIA 专属 CUDA 扩展；
- 标准 Triton 自定义 Kernel；
- FlashAttention；
- `fla-core`、`causal-conv1d` 等 GDN fast path 依赖；
- 在 NVIDIA GPU 上生成的 AWQ 性能结论。

对当前各方向的初步影响如下：

| 方向 | PPU 迁移判断 |
|---|---|
| A：ToMe 视觉合并 | 仅依赖 PyTorch，预期迁移风险最低，但仍需目标硬件复测 |
| C：KV Cache / Triton | 需要适配或验证 PPU Triton，不能直接沿用 NVIDIA 结果 |
| D：GDN CUDA | 当前显著收益来自 NVIDIA CUDA fast path，PPU 上仍属未验证 |
| AWQ | 算法和权重可能可迁移，但框架版本、运行时和实际性能需要重新验证 |

## 4. 公共服务器风险

公共高权限账号意味着其他参赛者理论上可以读取同一服务器中的源码、命令历史、
配置和实验结果。团队不应在该服务器上存放：

- GitHub Token、SSH 私钥或其他私人凭据；
- 团队完整仓库和未公开优化代码；
- 最终参数组合、实验结果和技术报告；
- 含有团队实现的虚拟环境、编译缓存或模型产物。

这台服务器适合检查公开硬件环境、SDK 和官方 baseline，不适合作为团队日常开发或
保密实验环境。

## 5. 建议

1. 继续在本地完成代码开发、测试和实验记录。
2. 所有优化保持默认关闭，并提供独立开关和清晰的 baseline 回退路径。
3. 公共服务器仅运行学校提供的公开程序和通用环境检查。
4. 在学校提供每队独立容器、私有账号或正式提交窗口后，再部署团队实现。
5. 在同一私有 PPU 环境和统一评测协议下重新比较 A、C、D 方向，不能直接合并
   Mac、NVIDIA GPU 和 PPU 上的性能百分比。

## 6. 当前结论

学校服务器具备充足的 PPU 显存、CPU 和系统内存，硬件资源不是 Qwen3.5-2B 单模型
推理的主要限制。当前最大问题是公共访问带来的代码保密风险，以及 NVIDIA CUDA
优化在 PPU 兼容层上的不确定性。

在获得隔离环境前，可以确认硬件和公开软件栈已经就绪，但不能安全地开展团队优化代码
的完整部署与性能复测。
