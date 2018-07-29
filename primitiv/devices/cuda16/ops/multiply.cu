#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDA16_KERNEL_FW_X_CONST(multiply_const, X_VAL * k);
CUDA16_KERNEL_BW_X_CONST(multiply_const, k * GY_VAL);

CUDA16_KERNEL_FW_X_SCALAR_R(multiply_scalar, ::__fmul_rn);

CUDA16_KERNEL_FW_AB(multiply, ::__fmul_rn);

DECL_ATOMIC_OP(atomicHAdd, ::__fadd_rn);

__global__ void multiply_bw_dev(
    const half *pa, const half *pb, const half *, const half *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb,
    half *pga, half *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const float gy = ::__half2float(pgy[i + shift]);
    const std::uint32_t a_ofs = i + mba * shift;
    const std::uint32_t b_ofs = i + mbb * shift;
    ::atomicHAdd(pga, a_ofs, gy * ::__half2float(pb[b_ofs]));
    ::atomicHAdd(pgb, b_ofs, gy * ::__half2float(pa[a_ofs]));
  }
}

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X_CONST(multiply_const);
CUDA16_DEV_BW_X_CONST(multiply_const);

CUDA16_DEV_FW_X_SCALAR(multiply_scalar);

CUDA16_DEV_FW_AB(multiply);
CUDA16_DEV_BW_AB(multiply);

}  // namespace devices
}  // namespace primitiv
