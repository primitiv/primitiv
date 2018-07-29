#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDADEV_KERNEL_FW_X_CONST(add_const, px[i] + k);
CUDADEV_KERNEL_BW_X_CONST(add_const, pgy[i]);

CUDADEV_KERNEL_FW_X_SCALAR_R(add_scalar, ::__fadd_rn);

CUDADEV_KERNEL_FW_AB(add, ::__fadd_rn);

__global__ void add_bw_dev(
    const float *, const float *, const float *, const float *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb,
    float *pga, float *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    ::atomicAdd(pga + i + mba * shift, gy);
    ::atomicAdd(pgb + i + mbb * shift, gy);
  }
}

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X_CONST(add_const);
CUDADEV_BW_X_CONST(add_const);

CUDADEV_FW_X_SCALAR(add_scalar);

CUDADEV_FW_AB(add);
CUDADEV_BW_AB(add);

}  // namespace devices
}  // namespace primitiv
