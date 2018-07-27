#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void concat_fw_dev(
    const half *px,
    std::uint32_t span, std::uint32_t skip, std::uint32_t x_size,
    std::uint32_t y_size, half *py) {
  const std::uint32_t i = IDX;
  if (i < y_size) py[(i / span) * skip + (i % span)] = px[i % x_size];
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::concat_fw_impl(
    const std::vector<const Tensor *> &xs, std::uint32_t dim, Tensor &y) {
  const std::uint32_t new_bs = y.shape().batch();
  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t skip = base * y.shape()[dim];
  std::uint32_t repeat = y.shape().volume() / skip;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  std::uint32_t offset = 0;
  for (const Tensor *x : xs) {
    const std::uint32_t span = base * x->shape()[dim];
    const std::uint32_t x_size = span * repeat * x->shape().batch();
    const std::uint32_t y_size = span * repeat * new_bs;
    const std::uint32_t num_blocks = GRID_SIZE(y_size, dim1_x_);
    ::concat_fw_dev<<<num_blocks, dim1_x_>>>(
       CDATA(half, *x), span, skip, x_size, y_size, MDATA(half, y) + offset);
    offset += span;
  }
}

}  // namespace devices
}  // namespace primitiv
