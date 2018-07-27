#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDA16_KERNEL_FW_X_CONST(divide_const_r, X_VAL / k);
CUDA16_KERNEL_BW_X_CONST(divide_const_r, GY_VAL / k);

CUDA16_KERNEL_FW_X_SCALAR_R(divide_scalar_r, ::__fdiv_rn);

CUDA16_KERNEL_FW_X_CONST(divide_const_l, k / X_VAL);
CUDA16_KERNEL_BW_X_CONST(divide_const_l, -Y_VAL * GY_VAL / X_VAL);

CUDA16_KERNEL_FW_X_SCALAR_L(divide_scalar_l, ::__fdiv_rn);

CUDA16_KERNEL_FW_AB(divide, ::__fdiv_rn);

DECL_ATOMIC_OP(atomicHAdd, ::__fadd_rn);

__global__ void divide_bw_dev(
    const half *, const half *pb, const half *py, const half *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb,
    half *pga, half *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const std::uint32_t b_ofs = i + mbb * shift;
    const std::uint32_t y_ofs = i + shift;
    const float k = ::__half2float(pgy[y_ofs]) / ::__half2float(pb[b_ofs]);
    ::atomicHAdd(pga, i + mba * shift, k);
    ::atomicHAdd(pgb, b_ofs, -k * __half2float(py[y_ofs]));
  }
}

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X_CONST(divide_const_r);
CUDA16_DEV_BW_X_CONST(divide_const_r);

CUDA16_DEV_FW_X_CONST(divide_const_l);
CUDA16_DEV_BW_X_CONST(divide_const_l);

CUDA16_DEV_FW_X_SCALAR(divide_scalar_r);

CUDA16_DEV_FW_X_SCALAR(divide_scalar_l);

CUDA16_DEV_FW_AB(divide);
CUDA16_DEV_BW_AB(divide);

}  // namespace devices
}  // namespace primitiv
