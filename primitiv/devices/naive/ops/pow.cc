#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X_CONST(pow_const_r, std::pow(src[i], k));
CPUDEV_BW_X_CONST(pow_const_r, k * pgy[i] * py[i] / px[i]);

CPUDEV_FW_X_CONST(pow_const_l, std::pow(k, src[i]));
CPUDEV_BW_X_CONST(pow_const_l, std::log(k) * pgy[i] * py[i]);

CPUDEV_FW_X_SCALAR(pow_scalar_r, std::pow(src_x[i], *src_k));

CPUDEV_FW_X_SCALAR(pow_scalar_l, std::pow(*src_k, src_x[i]));

CPUDEV_FW_AB(pow, std::pow(src_a[i], src_b[i]));

void Naive::pow_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  const std::uint32_t size = gy.shape().volume();
  const std::uint32_t bs = gy.shape().batch();
  const std::uint32_t skip_a = ga.shape().has_batch() * size;
  const std::uint32_t skip_b = gb.shape().has_batch() * size;
  const float *pa = CDATA(a);
  const float *pb = CDATA(b);
  const float *py = CDATA(y);
  const float *pgy = CDATA(gy);
  float *pga = MDATA(ga);
  float *pgb = MDATA(gb);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    for (std::uint32_t i = 0; i < size; ++i) {
      const float a = pgy[i] * py[i];
      pga[i] += a * pb[i] / pa[i];
      pgb[i] += a * std::log(pa[i]);
    }
    pa += skip_a;
    pb += skip_b;
    py += size;
    pgy += size;
    pga += skip_a;
    pgb += skip_b;
  }
}

}  // namespace devices
}  // namespace primitiv
