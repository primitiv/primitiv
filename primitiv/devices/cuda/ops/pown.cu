#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__device__ float pown_fw_element_dev(float x, std::int32_t k) {
  // NOTE(odashi):
  // std::abs(-0x80000000) is UB under 2's complement systems.
  // However, this value should be also evaluated as  0x80000000 by directly
  // casting to std::uint32_t.
  const std::int32_t min_k = -0x80000000;
  std::uint32_t remain = (k == min_k) ? min_k : ::abs(k);
  float ret = 1.f;
  float factor = x;

  // Performs the exponentation-by-squaring method.
  while (remain) {
    if (remain & 1) ret *= factor;
    factor *= factor;
    remain >>= 1;
  }

  return k >= 0 ? ret : 1.f / ret;
}

__global__ void pown_fw_dev(
    const float *px, std::int32_t k, std::uint32_t size, float *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = pown_fw_element_dev(px[i], k);
}

__global__ void pown_bw_dev(
    const float *px, const float *py, const float *pgy, std::int32_t k,
    std::uint32_t size, float *pgx) {
  static_cast<void>(px);
  static_cast<void>(py);
  const std::uint32_t i = IDX;
  if (i < size) pgx[i] += k * pgy[i] * py[i] / px[i];
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA::pown_fw_impl(const Tensor &x, std::int32_t k, Tensor &y) {
  const std::uint32_t size = x.shape().size();
  const std::uint32_t num_blocks = GRID_SIZE(size,dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::pown_fw_dev<<<num_blocks, dim1_x_>>>(CDATA(x), k, size, MDATA(y));
}

void CUDA::pown_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, std::int32_t k,
    Tensor &gx) {
  const std::uint32_t size = x.shape().size();
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::pown_bw_dev<<<num_blocks, dim1_x_>>>(
      CDATA(x), CDATA(y), CDATA(gy), k, size, MDATA(gx));
}

}  // namespace devices
}  // namespace primitiv
