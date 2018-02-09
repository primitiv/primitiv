#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16DEV_KERNEL_FW_X_CONST(subtract_const_r, px[i] - k);
CUDA16DEV_KERNEL_BW_X_CONST(subtract_const_r, pgy[i]);

CUDA16DEV_KERNEL_FW_X_SCALAR_R(subtract_scalar_r, ::__fsub_rn);

CUDA16DEV_KERNEL_FW_X_CONST(subtract_const_l, k - px[i]);
CUDA16DEV_KERNEL_BW_X_CONST(subtract_const_l, -pgy[i]);

CUDA16DEV_KERNEL_FW_X_SCALAR_L(subtract_scalar_l, ::__fsub_rn);

CUDA16DEV_KERNEL_FW_AB(subtract, ::__fsub_rn);

/*
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
*/

}  // namespace

namespace primitiv {
namespace devices {

CUDA16DEV_FW_X_CONST(subtract_const_r);
CUDA16DEV_BW_X_CONST(subtract_const_r);

CUDA16DEV_FW_X_CONST(subtract_const_l);
CUDA16DEV_BW_X_CONST(subtract_const_l);

CUDA16DEV_FW_X_SCALAR(subtract_scalar_r);

CUDA16DEV_FW_X_SCALAR(subtract_scalar_l);

CUDA16DEV_FW_AB(subtract);
CUDA16DEV_BW_AB(subtract);

}  // namespace devices
}  // namespace primitiv
