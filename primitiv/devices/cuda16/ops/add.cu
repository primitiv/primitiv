#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDA16_KERNEL_FW_X_CONST(add_const, X_VAL + k);
CUDA16_KERNEL_BW_X_CONST(add_const, GY_VAL);

CUDA16_KERNEL_FW_X_SCALAR_R(add_scalar, ::__fadd_rn);

CUDA16_KERNEL_FW_AB(add, ::__fadd_rn);

DECL_ATOMIC_OP(atomicHAdd, ::__fadd_rn);

__global__ void add_bw_dev(
    const half *, const half *, const half *, const half *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb,
    half *pga, half *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const float gy = ::__half2float(pgy[i + shift]);
    atomicHAdd(pga, i + mba * shift, gy);
    atomicHAdd(pgb, i + mbb * shift, gy);
  }
}

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X_CONST(add_const);
CUDA16_DEV_BW_X_CONST(add_const);

CUDA16_DEV_FW_X_SCALAR(add_scalar);

CUDA16_DEV_FW_AB(add);
CUDA16_DEV_BW_AB(add);

}  // namespace devices
}  // namespace primitiv
