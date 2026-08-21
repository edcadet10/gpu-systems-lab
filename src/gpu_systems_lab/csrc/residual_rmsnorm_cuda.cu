#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr int kThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kWarpCount = kThreads / kWarpSize;

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__global__ void residual_rmsnorm_fp16_kernel(
    const __half* input,
    const __half* residual,
    const __half* weight,
    __half* output,
    std::int64_t hidden_size,
    float epsilon) {
  const auto row = static_cast<std::int64_t>(blockIdx.x);
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const auto row_offset = row * hidden_size;

  float local_square_sum = 0.0f;
  float local_correction = 0.0f;
  for (std::int64_t column = threadIdx.x; column < hidden_size; column += blockDim.x) {
    const float summed = __half2float(input[row_offset + column])
        + __half2float(residual[row_offset + column]);
    const float corrected_square = summed * summed - local_correction;
    const float updated_sum = local_square_sum + corrected_square;
    local_correction = (updated_sum - local_square_sum) - corrected_square;
    local_square_sum = updated_sum;
  }

  local_square_sum = warp_sum(local_square_sum);
  __shared__ float warp_sums[kWarpCount];
  __shared__ float inverse_rms;
  if (lane == 0) {
    warp_sums[warp] = local_square_sum;
  }
  __syncthreads();

  if (warp == 0) {
    float block_square_sum = lane < kWarpCount ? warp_sums[lane] : 0.0f;
    block_square_sum = warp_sum(block_square_sum);
    if (lane == 0) {
      inverse_rms = rsqrtf(block_square_sum / static_cast<float>(hidden_size) + epsilon);
    }
  }
  __syncthreads();

  for (std::int64_t column = threadIdx.x; column < hidden_size; column += blockDim.x) {
    const float summed = __half2float(input[row_offset + column])
        + __half2float(residual[row_offset + column]);
    const float normalized = summed * inverse_rms * __half2float(weight[column]);
    output[row_offset + column] = __float2half_rn(normalized);
  }
}

torch::Tensor residual_rmsnorm_cuda(
    const torch::Tensor& input,
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    double epsilon) {
  TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  TORCH_CHECK(residual.is_cuda() && weight.is_cuda(), "all tensors must be CUDA tensors");
  TORCH_CHECK(input.scalar_type() == at::kHalf, "input must use float16");
  TORCH_CHECK(
      residual.scalar_type() == at::kHalf && weight.scalar_type() == at::kHalf,
      "all tensors must use float16");
  TORCH_CHECK(input.sizes() == residual.sizes(), "input and residual shapes must match");
  TORCH_CHECK(input.dim() >= 1, "input must have at least one dimension");
  TORCH_CHECK(weight.dim() == 1, "weight must be one-dimensional");
  TORCH_CHECK(weight.size(0) == input.size(-1), "weight must match the final dimension");
  TORCH_CHECK(
      input.is_contiguous() && residual.is_contiguous() && weight.is_contiguous(),
      "all tensors must be contiguous");
  TORCH_CHECK(input.get_device() == residual.get_device(), "input and residual devices differ");
  TORCH_CHECK(input.get_device() == weight.get_device(), "input and weight devices differ");
  TORCH_CHECK(input.numel() > 0, "input must be non-empty");
  TORCH_CHECK(std::isfinite(epsilon) && epsilon > 0.0, "epsilon must be positive and finite");
  TORCH_CHECK(
      std::isfinite(static_cast<float>(epsilon)), "epsilon must be representable as float32");

  const auto hidden_size = input.size(-1);
  const auto row_count = input.numel() / hidden_size;
  TORCH_CHECK(
      row_count <= std::numeric_limits<int>::max(), "row count exceeds the CUDA grid limit");

  c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty_like(input);
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream(input.get_device());
  residual_rmsnorm_fp16_kernel<<<static_cast<int>(row_count), kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(input.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(residual.data_ptr<at::Half>()),
      reinterpret_cast<const __half*>(weight.data_ptr<at::Half>()),
      reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
      hidden_size,
      static_cast<float>(epsilon));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("forward", &residual_rmsnorm_cuda, "Fused residual plus RMSNorm (CUDA, FP16)");
}
