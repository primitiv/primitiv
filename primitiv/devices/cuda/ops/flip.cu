#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/internal/cuda/utils.h>
#include <primitiv/devices/cuda/ops/common.h>

namespace {

__global__ void flip_fw_dev(
    const float *px, std::uint32_t skip, std::uint32_t n, std::uint32_t r, float *py) {
  const unsigned i = IDX;
  const unsigned j = IDY;
  const unsigned offset = j * n - j % skip * (n - 1);
  if (i < n && j < r) {
    py[offset + i * skip] = px[offset + (n - i - 1) * skip];
  }
}

__global__ void flip_bw_dev(
    const float *pgy, std::uint32_t skip, std::uint32_t n, std::uint32_t r, float *pgx) {
  const unsigned i = IDX;
  const unsigned j = IDY;
  const unsigned offset = j * n - j % skip * (n - 1);
  if (i < n && j < r) {
    pgx[offset + i * skip] += pgy[offset + (n - i - 1) * skip];
  }
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA::flip_fw_impl(
    const Tensor &x, std::uint32_t dim, Tensor &y) {
  const Shape &s = x.shape();
  const std::uint32_t n = s[dim];
  const std::uint32_t skip = s.lower_volume(dim);
  const std::uint32_t r = s.size() / n;
  const std::uint32_t g1 = GRID_SIZE(n, dim2_x_);
  const std::uint32_t g2 = GRID_SIZE(r, dim2_y_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::flip_fw_dev<<<dim3(g1, g2, 1), dim3(dim2_x_, dim2_y_, 1)>>>(
      CDATA(x), skip, n, r, MDATA(y));
}

void CUDA::flip_bw_impl(
    const Tensor &gy, std::uint32_t dim, Tensor &gx) {
  const Shape &s = gy.shape();
  const std::uint32_t n = s[dim];
  const std::uint32_t skip = s.lower_volume(dim);
  const std::uint32_t r = s.size() / n;
  const std::uint32_t g1 = GRID_SIZE(n, dim2_x_);
  const std::uint32_t g2 = GRID_SIZE(r, dim2_y_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::flip_bw_dev<<<dim3(g1, g2, 1), dim3(dim2_x_, dim2_y_, 1)>>>(
      CDATA(gy), skip, n, r, MDATA(gx));
}

}  // namespace devices
}  // namespace primitiv
