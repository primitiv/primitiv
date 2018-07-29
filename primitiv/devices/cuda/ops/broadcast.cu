#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void broadcast_fw_dev(
    const float *px,
    std::uint32_t skip1, std::uint32_t skip2, std::uint32_t size,
    float *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = px[i % skip1 + (i / skip2) * skip1];
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA::broadcast_fw_impl(
    const Tensor &x, std::uint32_t dim, std::uint32_t size, Tensor &y) {
  const std::uint32_t skip1 = y.shape().lower_volume(dim);
  const std::uint32_t skip2 = skip1 * size;
  const std::uint32_t total = y.shape().size();
  const std::uint32_t g1 = GRID_SIZE(total, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::broadcast_fw_dev<<<g1, dim1_x_>>>(CDATA(x), skip1, skip2, total, MDATA(y));
}

}  // namespace devices
}  // namespace primitiv
