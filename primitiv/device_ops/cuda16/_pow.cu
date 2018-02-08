#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16DEV_KERNEL_FW_X_CONST(pow_const_r, ::powf(px[i], k));
CUDA16DEV_KERNEL_BW_X_CONST(pow_const_r, k * pgy[i] * py[i] / px[i]);

CUDA16DEV_KERNEL_FW_X_SCALAR_R(pow_scalar_r, ::powf);

CUDA16DEV_KERNEL_FW_X_CONST(pow_const_l, ::powf(k, px[i]));
CUDA16DEV_KERNEL_BW_X_CONST(pow_const_l, ::logf(k) * pgy[i] * py[i]);

CUDA16DEV_KERNEL_FW_X_SCALAR_L(pow_scalar_l, ::powf);

CUDA16DEV_KERNEL_FW_AB(pow, ::powf);

__global__ void pow_bw_dev(
    const float *pa, const float *pb, const float *py, const float *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb,
    float *pga, float *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const std::uint32_t a_ofs = i + mba * shift;
    const std::uint32_t b_ofs = i + mbb * shift;
    const std::uint32_t y_ofs = i + shift;
    const float k = pgy[y_ofs] * py[y_ofs];
    ::atomicAdd(pga + a_ofs, k * pb[b_ofs] / pa[a_ofs]);
    ::atomicAdd(pgb + b_ofs, k * ::logf(pa[a_ofs]));
  }
}

}  // namespace

namespace primitiv {
namespace devices {

CUDA16DEV_FW_X_CONST(pow_const_r);
CUDA16DEV_BW_X_CONST(pow_const_r);

CUDA16DEV_FW_X_CONST(pow_const_l);
CUDA16DEV_BW_X_CONST(pow_const_l);

CUDA16DEV_FW_X_SCALAR(pow_scalar_r);

CUDA16DEV_FW_X_SCALAR(pow_scalar_l);

CUDA16DEV_FW_AB(pow);
CUDA16DEV_BW_AB(pow);

}  // namespace devices
}  // namespace primitiv
