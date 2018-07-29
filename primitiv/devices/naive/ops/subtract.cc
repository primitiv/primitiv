#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X_CONST(subtract_const_r, src[i] - k);
CPUDEV_BW_X_CONST(subtract_const_r, pgy[i]);

CPUDEV_FW_X_CONST(subtract_const_l, k - src[i]);
CPUDEV_BW_X_CONST(subtract_const_l, -pgy[i]);

CPUDEV_FW_X_SCALAR(subtract_scalar_r, src_x[i] - *src_k);

CPUDEV_FW_X_SCALAR(subtract_scalar_l, *src_k - src_x[i]);

CPUDEV_FW_AB(subtract, src_a[i] - src_b[i]);

void Naive::subtract_bw_impl(
    const Tensor &, const Tensor &, const Tensor &, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  const std::uint32_t size = gy.shape().volume();
  const std::uint32_t bs = gy.shape().batch();
  const std::uint32_t skip_a = ga.shape().has_batch() * size;
  const std::uint32_t skip_b = gb.shape().has_batch() * size;
  const float *pgy = CDATA(gy);
  float *pga = MDATA(ga);
  float *pgb = MDATA(gb);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    for (std::uint32_t i = 0; i < size; ++i) {
      const float k = pgy[i];
      pga[i] += k;
      pgb[i] -= k;
    }
    pgy += size;
    pga += skip_a;
    pgb += skip_b;
  }
}

}  // namespace devices
}  // namespace primitiv
