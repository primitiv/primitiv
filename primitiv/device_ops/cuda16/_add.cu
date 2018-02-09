#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16DEV_KERNEL_FW_X_CONST(add_const, px[i] + k);
CUDA16DEV_KERNEL_BW_X_CONST(add_const, pgy[i]);

CUDA16DEV_KERNEL_FW_X_SCALAR_R(add_scalar, ::__fadd_rn);

CUDA16DEV_KERNEL_FW_AB(add, ::__fadd_rn);

/*
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
*/

}  // namespace

namespace primitiv {
namespace devices {

CUDA16DEV_FW_X_CONST(add_const);
CUDA16DEV_BW_X_CONST(add_const);

CUDA16DEV_FW_X_SCALAR(add_scalar);

CUDA16DEV_FW_AB(add);
CUDA16DEV_BW_AB(add);

}  // namespace devices
}  // namespace primitiv
