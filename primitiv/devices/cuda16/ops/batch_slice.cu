#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void batch_slice_fw_dev(
    const half *px, std::uint32_t size, half *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = px[i];
}

__global__ void batch_slice_bw_dev(
    const half *pgy, std::uint32_t size, half *pgx) {
  const std::uint32_t i = IDX;
  if (i < size) {
    INPLACE_ADD(pgx + i, ::__half2float(pgy[i]));
  }
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::batch_slice_fw_impl(
    const Tensor &x, std::uint32_t offset, Tensor &y) {
  const std::uint32_t volume = y.shape().volume();
  const std::uint32_t size = y.shape().size();
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_);

  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::batch_slice_fw_dev<<<num_blocks, dim1_x_>>>(
      CDATA(half, x) + volume * offset, size, MDATA(half, y));
}

void CUDA16::batch_slice_bw_impl(
    const Tensor &gy, std::uint32_t offset, Tensor &gx) {
  const std::uint32_t volume = gy.shape().volume();
  const std::uint32_t size = gy.shape().size();
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_);

  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::batch_slice_bw_dev<<<num_blocks, dim1_x_>>>(
      CDATA(half, gy), size, MDATA(half, gx) + volume * offset);
}

}  // namespace devices
}  // namespace primitiv
