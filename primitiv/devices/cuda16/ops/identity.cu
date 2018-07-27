#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void set_identity_dev(
    std::uint32_t size, std::uint32_t skip, half *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = ::__float2half(!(i % skip));
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::identity_impl(Tensor &y) {
  const std::uint32_t size = y.shape().size();
  const std::uint32_t skip = y.shape()[0] + 1;
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::set_identity_dev<<<num_blocks, dim1_x_>>>(size, skip, MDATA(half, y));
}

}  // namespace devices
}  // namespace primitiv
