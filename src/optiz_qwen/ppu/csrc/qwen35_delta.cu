#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

namespace {

constexpr int kHeadDim = 128;
constexpr int kWarpsPerBlock = kHeadDim / 32;

__device__ double block_sum(double value, double* warp_sums) {
  constexpr unsigned int mask = 0xffffffffu;
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(mask, value, offset);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) {
    warp_sums[warp] = value;
  }
  __syncthreads();
  if (warp == 0) {
    value = lane < kWarpsPerBlock ? warp_sums[lane] : 0.0;
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(mask, value, offset);
    }
    if (lane == 0) {
      warp_sums[0] = value;
    }
  }
  __syncthreads();
  return warp_sums[0];
}

__global__ void qwen35_delta_kernel(
    const __nv_bfloat16* query,
    const __nv_bfloat16* key,
    const __nv_bfloat16* value,
    const float* g,
    const __nv_bfloat16* beta,
    __nv_bfloat16* output,
    float* final_state,
    int sequence,
    int heads,
    int value_dim) {
  const int batch_index = blockIdx.z;
  const int head_index = blockIdx.x;
  const int value_index = blockIdx.y;
  const int key_index = threadIdx.x;
  __shared__ double warp_sums[kWarpsPerBlock];

  float state = 0.0f;
  constexpr float scale = 0.08838834764831845f;
  for (int token_index = 0; token_index < sequence; ++token_index) {
    const int qk_offset =
        ((batch_index * sequence + token_index) * heads + head_index) * kHeadDim + key_index;
    const int value_offset =
        ((batch_index * sequence + token_index) * heads + head_index) * value_dim + value_index;
    const int gate_offset =
        (batch_index * sequence + token_index) * heads + head_index;
    const float key_element = __bfloat162float(key[qk_offset]);
    state *= expf(g[gate_offset]);
    const double memory = block_sum(
        static_cast<double>(state) * static_cast<double>(key_element),
        warp_sums);
    const double delta =
        (__bfloat162float(value[value_offset]) - memory) * __bfloat162float(beta[gate_offset]);
    state = static_cast<float>(static_cast<double>(state) + key_element * delta);
    const double result = block_sum(
        static_cast<double>(state) * __bfloat162float(query[qk_offset]) * scale,
        warp_sums);
    if (key_index == 0) {
      output[value_offset] = __float2bfloat16_rn(result);
    }
  }
  const int state_offset =
      ((batch_index * heads + head_index) * kHeadDim + key_index) * value_dim + value_index;
  final_state[state_offset] = state;
}

void require_same_device(const torch::Tensor& reference, const torch::Tensor& value) {
  TORCH_CHECK(value.device() == reference.device(), "all Delta Rule inputs must share one device");
}

}  // namespace

std::vector<torch::Tensor> qwen35_delta_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta) {
  TORCH_CHECK(query.is_cuda(), "query must be on PPU");
  require_same_device(query, key);
  require_same_device(query, value);
  require_same_device(query, g);
  require_same_device(query, beta);
  TORCH_CHECK(query.scalar_type() == torch::kBFloat16, "query must be bfloat16");
  TORCH_CHECK(key.scalar_type() == torch::kBFloat16, "key must be bfloat16");
  TORCH_CHECK(value.scalar_type() == torch::kBFloat16, "value must be bfloat16");
  TORCH_CHECK(beta.scalar_type() == torch::kBFloat16, "beta must be bfloat16");
  TORCH_CHECK(g.scalar_type() == torch::kFloat32, "g must be float32");
  TORCH_CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous(),
              "query, key, and value must be contiguous");
  TORCH_CHECK(g.is_contiguous() && beta.is_contiguous(), "g and beta must be contiguous");
  TORCH_CHECK(query.dim() == 4 && query.sizes() == key.sizes(), "invalid query/key shapes");
  TORCH_CHECK(value.dim() == 4 && value.size(0) == query.size(0) &&
              value.size(1) == query.size(1) && value.size(2) == query.size(2),
              "invalid value shape");
  TORCH_CHECK(query.size(3) == kHeadDim, "key head dimension must be 128");
  TORCH_CHECK(value.size(3) == kHeadDim, "value head dimension must be 128");
  TORCH_CHECK(g.sizes() == beta.sizes(), "g and beta shapes must match");
  TORCH_CHECK(g.dim() == 3 && g.size(0) == query.size(0) &&
              g.size(1) == query.size(1) && g.size(2) == query.size(2),
              "invalid gate shape");

  c10::cuda::CUDAGuard guard(query.device());
  auto output = torch::empty_like(value);
  auto state = torch::empty(
      {query.size(0), query.size(2), kHeadDim, value.size(3)},
      query.options().dtype(torch::kFloat32));
  const dim3 grid(query.size(2), value.size(3), query.size(0));
  const auto stream = at::cuda::getCurrentCUDAStream(query.device().index());
  qwen35_delta_kernel<<<grid, kHeadDim, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(query.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(key.data_ptr<at::BFloat16>()),
      reinterpret_cast<const __nv_bfloat16*>(value.data_ptr<at::BFloat16>()),
      g.data_ptr<float>(),
      reinterpret_cast<const __nv_bfloat16*>(beta.data_ptr<at::BFloat16>()),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
      state.data_ptr<float>(),
      query.size(1),
      query.size(2),
      value.size(3));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output, state};
}
