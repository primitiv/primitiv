#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDADEV_KERNEL_FW_X_CONST(pow_const_r, ::powf(px[i], k));
CUDADEV_KERNEL_BW_X_CONST(pow_const_r, k * pgy[i] * py[i] / px[i]);

CUDADEV_KERNEL_FW_X_SCALAR_R(pow_scalar_r, ::powf);

CUDADEV_KERNEL_FW_X_CONST(pow_const_l, ::powf(k, px[i]));
CUDADEV_KERNEL_BW_X_CONST(pow_const_l, ::logf(k) * pgy[i] * py[i]);

CUDADEV_KERNEL_FW_X_SCALAR_L(pow_scalar_l, ::powf);

CUDADEV_KERNEL_FW_AB(pow, ::powf);

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

CUDADEV_FW_X_CONST(pow_const_r);
CUDADEV_BW_X_CONST(pow_const_r);

CUDADEV_FW_X_CONST(pow_const_l);
CUDADEV_BW_X_CONST(pow_const_l);

CUDADEV_FW_X_SCALAR(pow_scalar_r);

CUDADEV_FW_X_SCALAR(pow_scalar_l);

CUDADEV_FW_AB(pow);
CUDADEV_BW_AB(pow);

}  // namespace devices
}  // namespace primitiv
