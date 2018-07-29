#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDADEV_KERNEL_FW_X_CONST(subtract_const_r, px[i] - k);
CUDADEV_KERNEL_BW_X_CONST(subtract_const_r, pgy[i]);

CUDADEV_KERNEL_FW_X_SCALAR_R(subtract_scalar_r, ::__fsub_rn);

CUDADEV_KERNEL_FW_X_CONST(subtract_const_l, k - px[i]);
CUDADEV_KERNEL_BW_X_CONST(subtract_const_l, -pgy[i]);

CUDADEV_KERNEL_FW_X_SCALAR_L(subtract_scalar_l, ::__fsub_rn);

CUDADEV_KERNEL_FW_AB(subtract, ::__fsub_rn);

__global__ void subtract_bw_dev(
    const float *, const float *, const float *, const float *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb,
    float *pga, float *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    ::atomicAdd(pga + i + mba * shift, gy);
    ::atomicAdd(pgb + i + mbb * shift, -gy);
  }
}

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X_CONST(subtract_const_r);
CUDADEV_BW_X_CONST(subtract_const_r);

CUDADEV_FW_X_CONST(subtract_const_l);
CUDADEV_BW_X_CONST(subtract_const_l);

CUDADEV_FW_X_SCALAR(subtract_scalar_r);

CUDADEV_FW_X_SCALAR(subtract_scalar_l);

CUDADEV_FW_AB(subtract);
CUDADEV_BW_AB(subtract);

}  // namespace devices
}  // namespace primitiv
