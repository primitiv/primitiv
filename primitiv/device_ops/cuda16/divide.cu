#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16DEV_KERNEL_FW_X_CONST(divide_const_r, px[i] / k);
CUDA16DEV_KERNEL_BW_X_CONST(divide_const_r, pgy[i] / k);

CUDA16DEV_KERNEL_FW_X_SCALAR_R(divide_scalar_r, ::__fdiv_rn);

CUDA16DEV_KERNEL_FW_X_CONST(divide_const_l, k / px[i]);
CUDA16DEV_KERNEL_BW_X_CONST(divide_const_l, -py[i] * pgy[i] / px[i]);

CUDA16DEV_KERNEL_FW_X_SCALAR_L(divide_scalar_l, ::__fdiv_rn);

CUDA16DEV_KERNEL_FW_AB(divide, ::__fdiv_rn);

__global__ void divide_bw_dev(
    const float *, const float *pb, const float *py, const float *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb,
    float *pga, float *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const std::uint32_t b_ofs = i + mbb * shift;
    const std::uint32_t y_ofs = i + shift;
    const float k = pgy[y_ofs] / pb[b_ofs];
    ::atomicAdd(pga + i + mba * shift, k);
    ::atomicAdd(pgb + b_ofs, -k * py[y_ofs]);
  }
}

}  // namespace

namespace primitiv {
namespace devices {

CUDA16DEV_FW_X_CONST(divide_const_r);
CUDA16DEV_BW_X_CONST(divide_const_r);

CUDA16DEV_FW_X_CONST(divide_const_l);
CUDA16DEV_BW_X_CONST(divide_const_l);

CUDA16DEV_FW_X_SCALAR(divide_scalar_r);

CUDA16DEV_FW_X_SCALAR(divide_scalar_l);

CUDA16DEV_FW_AB(divide);
CUDA16DEV_BW_AB(divide);

}  // namespace devices
}  // namespace primitiv
