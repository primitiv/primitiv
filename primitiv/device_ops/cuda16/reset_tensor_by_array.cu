#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/internal/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

__global__ void fp32to16(const float *src, half *dest, std::size_t size) {
  const std::size_t i = IDX;
  if (i < size) dest[i] = ::__float2half(src[i]);
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::reset_tensor_by_array_impl(const float values[], Tensor &x) {
  const std::size_t size = x.shape().size();
  const std::size_t gs = GRID_SIZE(size, dim1_x_);

  auto temp = state_->pool.allocate(sizeof(float) * size);
  float *temp_ptr = static_cast<float *>(temp.get());

  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        temp_ptr, values, sizeof(float) * size, cudaMemcpyHostToDevice));
  ::fp32to16<<<gs, dim1_x_>>>(temp_ptr, MDATA(half, x), size);
}

}  // namespace devices
}  // namespace primitiv
