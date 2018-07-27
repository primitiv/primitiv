#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDA16_KERNEL_FW_X_CONST(pow_const_r, ::powf(X_VAL, k));
CUDA16_KERNEL_BW_X_CONST(pow_const_r, k * GY_VAL * Y_VAL / X_VAL);

CUDA16_KERNEL_FW_X_SCALAR_R(pow_scalar_r, ::powf);

CUDA16_KERNEL_FW_X_CONST(pow_const_l, ::powf(k, X_VAL));
CUDA16_KERNEL_BW_X_CONST(pow_const_l, ::logf(k) * GY_VAL * Y_VAL);

CUDA16_KERNEL_FW_X_SCALAR_L(pow_scalar_l, ::powf);

CUDA16_KERNEL_FW_AB(pow, ::powf);

DECL_ATOMIC_OP(atomicHAdd, ::__fadd_rn);

__global__ void pow_bw_dev(
    const half *pa, const half *pb, const half *py, const half *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb,
    half *pga, half *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const std::uint32_t a_ofs = i + mba * shift;
    const std::uint32_t b_ofs = i + mbb * shift;
    const std::uint32_t y_ofs = i + shift;
    const float k = ::__half2float(pgy[y_ofs]) * ::__half2float(py[y_ofs]);
    const float pa_val = ::__half2float(pa[a_ofs]);
    ::atomicHAdd(pga, a_ofs, k * ::__half2float(pb[b_ofs]) / pa_val);
    ::atomicHAdd(pgb, b_ofs, k * ::logf(pa_val));
  }
}

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X_CONST(pow_const_r);
CUDA16_DEV_BW_X_CONST(pow_const_r);

CUDA16_DEV_FW_X_CONST(pow_const_l);
CUDA16_DEV_BW_X_CONST(pow_const_l);

CUDA16_DEV_FW_X_SCALAR(pow_scalar_r);

CUDA16_DEV_FW_X_SCALAR(pow_scalar_l);

CUDA16_DEV_FW_AB(pow);
CUDA16_DEV_BW_AB(pow);

}  // namespace devices
}  // namespace primitiv
