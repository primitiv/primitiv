#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__device__ half pown_fw_element_dev(half x, std::int32_t k) {
  // NOTE(odashi):
  // std::abs(-0x80000000) is UB under 2's complement systems.
  // However, this value should be also evaluated as  0x80000000 by directly
  // casting to std::uint32_t.
  const std::int32_t min_k = -0x80000000;
  std::uint32_t remain = (k == min_k) ? min_k : ::abs(k);
  float ret = 1.f;
  float factor = ::__half2float(x);

  // Performs the exponentation-by-squaring method.
  while (remain) {
    if (remain & 1) ret *= factor;
    factor *= factor;
    remain >>= 1;
  }

  return ::__float2half(k >= 0 ? ret : 1.f / ret);
}

__global__ void pown_fw_dev(
    const half *px, std::int32_t k, std::uint32_t size, half *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = pown_fw_element_dev(px[i], k);
}

__global__ void pown_bw_dev(
    const half *px, const half *py, const half *pgy, std::int32_t k,
    std::uint32_t size, half *pgx) {
  static_cast<void>(px);
  static_cast<void>(py);
  const std::uint32_t i = IDX;
  if (i < size) INPLACE_ADD(pgx + i, k * GY_VAL * Y_VAL / X_VAL);
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::pown_fw_impl(const Tensor &x, std::int32_t k, Tensor &y) {
  const std::uint32_t size = x.shape().size();
  const std::uint32_t num_blocks = GRID_SIZE(size,dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::pown_fw_dev<<<num_blocks, dim1_x_>>>(
      CDATA(half, x), k, size, MDATA(half, y));
}

void CUDA16::pown_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, std::int32_t k,
    Tensor &gx) {
  const std::uint32_t size = x.shape().size();
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::pown_bw_dev<<<num_blocks, dim1_x_>>>(
      CDATA(half, x), CDATA(half, y), CDATA(half, gy), k, size,
      MDATA(half, gx));
}

}  // namespace devices
}  // namespace primitiv
