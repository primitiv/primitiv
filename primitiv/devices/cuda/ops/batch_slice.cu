#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void batch_slice_fw_dev(
    const float *px, std::uint32_t size, float *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = px[i];
}

__global__ void batch_slice_bw_dev(
    const float *pgy, std::uint32_t size, float *pgx) {
  const std::uint32_t i = IDX;
  if (i < size) {
    pgx[i] += pgy[i];
  }
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA::batch_slice_fw_impl(
    const Tensor &x, std::uint32_t offset, Tensor &y) {
  const std::uint32_t volume = y.shape().volume();
  const std::uint32_t size = y.shape().size();
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_);

  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::batch_slice_fw_dev<<<num_blocks, dim1_x_>>>(
      CDATA(x) + volume * offset, size, MDATA(y));
}

void CUDA::batch_slice_bw_impl(
    const Tensor &gy, std::uint32_t offset, Tensor &gx) {
  const std::uint32_t volume = gy.shape().volume();
  const std::uint32_t size = gy.shape().size();
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_);

  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::batch_slice_bw_dev<<<num_blocks, dim1_x_>>>(
      CDATA(gy), size, MDATA(gx) + volume * offset);
}

}  // namespace devices
}  // namespace primitiv
