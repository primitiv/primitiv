#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void inplace_add_dev(
    const float *px,
    std::uint32_t size, std::uint32_t mbx, std::uint32_t mby,
    float *py) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) ::atomicAdd(py + i + mby * shift, px[i + mbx * shift]);
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA::inplace_add_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t size = y.shape().volume();
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_);
  const std::uint32_t bs = std::max(x.shape().batch(), y.shape().batch());
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::inplace_add_dev<<<dim3(g1, bs, 1), dim1_x_>>>(
      CDATA(x), size, x.shape().has_batch(), y.shape().has_batch(), MDATA(y));
}

}  // namespace devices
}  // namespace primitiv
