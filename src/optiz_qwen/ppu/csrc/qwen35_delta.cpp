#include <torch/extension.h>

#include <vector>

std::vector<torch::Tensor> qwen35_delta_forward(
    torch::Tensor query,
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor g,
    torch::Tensor beta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &qwen35_delta_forward, "Qwen3.5 PPU Delta Rule forward");
}
