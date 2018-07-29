#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X_CONST(divide_const_r, src[i] / k);
CPUDEV_BW_X_CONST(divide_const_r, pgy[i] / k);

CPUDEV_FW_X_CONST(divide_const_l, k / src[i]);
CPUDEV_BW_X_CONST(divide_const_l, -py[i] * pgy[i] / px[i]);

CPUDEV_FW_X_SCALAR(divide_scalar_r, src_x[i] / *src_k);

CPUDEV_FW_X_SCALAR(divide_scalar_l, *src_k / src_x[i]);

CPUDEV_FW_AB(divide, src_a[i] / src_b[i]);

void Naive::divide_bw_impl(
    const Tensor &, const Tensor &b, const Tensor &y, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  const std::uint32_t size = gy.shape().volume();
  const std::uint32_t bs = gy.shape().batch();
  const std::uint32_t skip_a = ga.shape().has_batch() * size;
  const std::uint32_t skip_b = gb.shape().has_batch() * size;
  const float *pb = CDATA(b);
  const float *py = CDATA(y);
  const float *pgy = CDATA(gy);
  float *pga = MDATA(ga);
  float *pgb = MDATA(gb);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    for (std::uint32_t i = 0; i < size; ++i) {
      const float k = pgy[i] / pb[i];
      pga[i] += k;
      pgb[i] -= k * py[i];
    }
    pb += skip_b;
    py += size;
    pgy += size;
    pga += skip_a;
    pgb += skip_b;
  }
}

}  // namespace devices
}  // namespace primitiv
