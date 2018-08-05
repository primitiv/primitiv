#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void batch_concat_fw_dev(
    const float *px, std::uint32_t y_size, float *py) {
  const std::uint32_t i = IDX;
  if (i < y_size) py[i] = px[i];
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA::batch_concat_fw_impl(
    const std::vector<const Tensor *> &xs, Tensor &y) {
  CUDA_CALL(::cudaSetDevice(dev_id_));
  std::uint32_t offset = 0;
  for (const Tensor *x : xs) {
    const std::uint32_t span = x->shape().size();
    const std::uint32_t num_blocks = GRID_SIZE(span, dim1_x_);
    ::batch_concat_fw_dev<<<num_blocks, dim1_x_>>>(
       CDATA(*x), span, MDATA(y) + offset);
    offset += span;
  }
}

}  // namespace devices
}  // namespace primitiv
