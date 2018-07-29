#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void fp16to32(const half *src, float *dest, std::size_t size) {
  const std::size_t i = IDX;
  if (i < size) dest[i] = ::__half2float(src[i]);
}

}  // namespace

namespace primitiv {
namespace devices {

std::vector<float> CUDA16::tensor_to_vector_impl(const Tensor &x) {
  const std::size_t size = x.shape().size();
  const std::size_t gs = GRID_SIZE(size, dim1_x_);

  auto temp = state_->pool.allocate(sizeof(float) * size);
  float *temp_ptr = static_cast<float *>(temp.get());
  std::vector<float> ret(size);

  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::fp16to32<<<gs, dim1_x_>>>(CDATA(half, x), temp_ptr, size);
  CUDA_CALL(::cudaMemcpy(
        ret.data(), temp_ptr, sizeof(float) * size, cudaMemcpyDeviceToHost));
  return ret;
}

}  // namespace devices
}  // namespace primitiv
