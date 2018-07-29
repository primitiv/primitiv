#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void slice_fw_dev(
    const float *px,
    std::uint32_t span, std::uint32_t skip, std::uint32_t size,
    float *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = px[(i / span) * skip + (i % span)];
}

__global__ void slice_bw_dev(
    const float *pgy,
    std::uint32_t wx, std::uint32_t wy, std::uint32_t nx, std::uint32_t ny,
    float *pgx) {
  const std::uint32_t i = IDX;
  if (i < wy * ::max(nx, ny)) {
    ::atomicAdd(
        pgx + ((i / wy) * wx + (i % wy)) % (wx * nx), pgy[i % (wy * ny)]);
  }
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA::slice_fw_impl(
    const Tensor &x, std::uint32_t dim, std::uint32_t offset, Tensor &y) {
  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t span = base * y.shape()[dim];
  const std::uint32_t skip = base * x.shape()[dim];
  const std::uint32_t size = y.shape().size();
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::slice_fw_dev<<<num_blocks, dim1_x_>>>(
      CDATA(x) + base * offset, span, skip, size, MDATA(y));
}

void CUDA::slice_bw_impl(
    const Tensor &gy, std::uint32_t dim, std::uint32_t offset, Tensor &gx) {
  const Shape &sx = gx.shape();
  const Shape &sy = gy.shape();
  const std::uint32_t base = sx.lower_volume(dim);
  const std::uint32_t ox = base * offset;
  const std::uint32_t wx = base * sx[dim];
  const std::uint32_t wy = base * sy[dim];
  const std::uint32_t repeat = sx.volume() / wx;
  const std::uint32_t nx = repeat * sx.batch();
  const std::uint32_t ny = repeat * sy.batch();
  const std::uint32_t g1 = GRID_SIZE(wy * std::max(nx, ny), dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::slice_bw_dev<<<g1, dim1_x_>>>(CDATA(gy), wx, wy, nx, ny, MDATA(gx) + ox);
}

}  // namespace devices
}  // namespace primitiv
