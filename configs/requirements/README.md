# Requirements Presets

Dependency snapshots for reproducible local benchmark flows live here.

- `dndx_public.txt`: organizer-compatible minimum runtime for the public DNDX self-test path
- `local_dev_extra.txt`: local-only helper packages for model download and development checks
- `local_gpu_windows_cuda.txt`: local Windows GPU wheel overlay for the `optiz-qwen` conda smoke-test environment
- `d_awq_cuda_py312.txt`: isolated Python 3.12 AWQ W4A16 preparation environment;
  install the matched Torch/TorchVision CUDA pair first
- `runtime_gdn_cuda_py312.txt`: optional Qwen3.5 GDN CUDA kernels; install them
  into a separate `--target` overlay so the baseline remains uncontaminated
