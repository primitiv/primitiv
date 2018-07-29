#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDA16_KERNEL_FW_X_CONST(subtract_const_r, X_VAL - k);
CUDA16_KERNEL_BW_X_CONST(subtract_const_r, GY_VAL);

CUDA16_KERNEL_FW_X_SCALAR_R(subtract_scalar_r, ::__fsub_rn);

CUDA16_KERNEL_FW_X_CONST(subtract_const_l, k - X_VAL);
CUDA16_KERNEL_BW_X_CONST(subtract_const_l, -GY_VAL);

CUDA16_KERNEL_FW_X_SCALAR_L(subtract_scalar_l, ::__fsub_rn);

CUDA16_KERNEL_FW_AB(subtract, ::__fsub_rn);

DECL_ATOMIC_OP(atomicHAdd, ::__fadd_rn);

__global__ void subtract_bw_dev(
    const half *, const half *, const half *, const half *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb,
    half *pga, half *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const float gy = ::__half2float(pgy[i + shift]);
    ::atomicHAdd(pga, i + mba * shift, gy);
    ::atomicHAdd(pgb, i + mbb * shift, -gy);
  }
}

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X_CONST(subtract_const_r);
CUDA16_DEV_BW_X_CONST(subtract_const_r);

CUDA16_DEV_FW_X_CONST(subtract_const_l);
CUDA16_DEV_BW_X_CONST(subtract_const_l);

CUDA16_DEV_FW_X_SCALAR(subtract_scalar_r);

CUDA16_DEV_FW_X_SCALAR(subtract_scalar_l);

CUDA16_DEV_FW_AB(subtract);
CUDA16_DEV_BW_AB(subtract);

}  // namespace devices
}  // namespace primitiv
