#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void rand_bernoulli_dev(float p, float size, float *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = (float)(py[i] <= p);
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA::random_bernoulli_impl(float p, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateUniform(state_->curand.get(), MDATA(y), size));
  ::rand_bernoulli_dev<<<num_blocks, dim1_x_>>>(p, size, MDATA(y));
}

}  // namespace devices
}  // namespace primitiv
