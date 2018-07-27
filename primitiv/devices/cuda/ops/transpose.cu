#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void transpose_fw_dev(
    const float *px, std::uint32_t rows, std::uint32_t cols, float *py) {
  const std::uint32_t i = IDX;
  const std::uint32_t j = IDY;
  std::uint32_t ofs = blockIdx.z * rows * cols;
  if (i < rows && j < cols) py[ofs + j + i * cols] = px[ofs + i + j * rows];
}

__global__ void transpose_bw_dev(
    const float *py, std::uint32_t rows, std::uint32_t cols, float *px) {
  const std::uint32_t i = IDX;
  const std::uint32_t j = IDY;
  std::uint32_t ofs = blockIdx.z * rows * cols;
  if (i < rows && j < cols) px[ofs + i + j * rows] += py[ofs + j + i * cols];
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA::transpose_fw_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t rows = x.shape()[0];
  const std::uint32_t cols = x.shape()[1];
  const std::uint32_t bs = x.shape().batch();
  const std::uint32_t g1 = GRID_SIZE(rows, dim2_x_);
  const std::uint32_t g2 = GRID_SIZE(cols, dim2_y_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::transpose_fw_dev<<<dim3(g1, g2, bs), dim3(dim2_x_, dim2_y_, 1)>>>(
      CDATA(x), rows, cols, MDATA(y));
}

void CUDA::transpose_bw_impl(
    const Tensor &, const Tensor &, const Tensor &gy, Tensor &gx) {
  const std::uint32_t rows = gx.shape()[0];
  const std::uint32_t cols = gx.shape()[1];
  const std::uint32_t bs = gx.shape().batch();
  const std::uint32_t g1 = GRID_SIZE(rows, dim2_x_);
  const std::uint32_t g2 = GRID_SIZE(cols, dim2_y_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::transpose_bw_dev<<<dim3(g1, g2, bs), dim3(dim2_x_, dim2_y_, 1)>>>(
      CDATA(gy), rows, cols, MDATA(gx));
}

}  // namespace devices
}  // namespace primitiv
