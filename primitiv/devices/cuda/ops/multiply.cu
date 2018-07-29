#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDADEV_KERNEL_FW_X_CONST(multiply_const, px[i] * k);
CUDADEV_KERNEL_BW_X_CONST(multiply_const, k * pgy[i]);

CUDADEV_KERNEL_FW_X_SCALAR_R(multiply_scalar, ::__fmul_rn);

CUDADEV_KERNEL_FW_AB(multiply, ::__fmul_rn);

__global__ void multiply_bw_dev(
    const float *pa, const float *pb, const float *, const float *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb,
    float *pga, float *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    const std::uint32_t a_ofs = i + mba * shift;
    const std::uint32_t b_ofs = i + mbb * shift;
    ::atomicAdd(pga + a_ofs, gy * pb[b_ofs]);
    ::atomicAdd(pgb + b_ofs, gy * pa[a_ofs]);
  }
}

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X_CONST(multiply_const);
CUDADEV_BW_X_CONST(multiply_const);

CUDADEV_FW_X_SCALAR(multiply_scalar);

CUDADEV_FW_AB(multiply);
CUDADEV_BW_AB(multiply);

}  // namespace devices
}  // namespace primitiv
