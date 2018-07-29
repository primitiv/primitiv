#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

DECL_ATOMIC_OP(atomicHAdd, ::__fadd_rn);

__global__ void inplace_add_dev(
    const half *px,
    std::uint32_t volume, std::uint32_t mbx, std::uint32_t mby,
    half *py) {
  const std::uint32_t i = IDX;

  if (i < volume) {
    const std::uint32_t offset = blockIdx.y * volume;
    const float x = ::__half2float(px[i + mbx * offset]);
    const std::uint32_t iy = i + mby * offset;
    ::atomicHAdd(py, iy, x);
  }
}

} // namespace

namespace primitiv {
namespace devices {

void CUDA16::inplace_add_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t volume = y.shape().volume();
  const std::uint32_t g1 = GRID_SIZE(volume, dim1_x_);
  const std::uint32_t bs = std::max(x.shape().batch(), y.shape().batch());

  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::inplace_add_dev<<<dim3(g1, bs, 1), dim1_x_>>>(
      CDATA(half, x),
      volume, x.shape().has_batch(), y.shape().has_batch(),
      MDATA(half, y));
}

}  // namespace devices
}  // namespace primitiv
