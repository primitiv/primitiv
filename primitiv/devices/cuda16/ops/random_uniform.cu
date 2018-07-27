#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void rand_affine_dev(
    const float *px, float shift, float scale, std::uint32_t size, half *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = ::__float2half(px[i] * scale + shift);
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::random_uniform_impl(float lower, float upper, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  const std::uint32_t gs = GRID_SIZE(size, dim1_x_);
  const float scale = upper - lower;
  auto temp = state_->pool.allocate(size * sizeof(float));
  float *temp_ptr = static_cast<float *>(temp.get());

  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateUniform(state_->curand.get(), temp_ptr, size));
  ::rand_affine_dev<<<gs, dim1_x_>>>(
      temp_ptr, lower, scale, size, MDATA(half, y));
}

}  // namespace devices
}  // namespace primitiv
