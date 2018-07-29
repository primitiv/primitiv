#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void rand_bernoulli_dev(
    const float *px, float p, float size, half *py) {
  // NOTE(odashi):
  // 0x0000 == 0.0
  // 0x3c00 == 1.0
  const std::size_t i = IDX;
  if (i < size) py[i] = ::__ushort_as_half(0x3c00 * (px[i] <= p));
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::random_bernoulli_impl(float p, Tensor &y) {
  const std::size_t size = y.shape().size();
  const std::size_t gs = GRID_SIZE(size, dim1_x_);
  auto temp = state_->pool.allocate(size * sizeof(float));
  float *temp_ptr = static_cast<float *>(temp.get());

  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateUniform(state_->curand.get(), temp_ptr, size));
  ::rand_bernoulli_dev<<<gs, dim1_x_>>>(temp_ptr, p, size, MDATA(half, y));
}

}  // namespace devices
}  // namespace primitiv
