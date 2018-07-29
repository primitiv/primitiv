#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X_CONST(add_const, src[i] + k);
CPUDEV_BW_X_CONST(add_const, pgy[i]);

CPUDEV_FW_X_SCALAR(add_scalar, src_x[i] + *src_k);

CPUDEV_FW_AB(add, src_a[i] + src_b[i]);

void Naive::add_bw_impl(
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
      pgb[i] += k;
    }
    pgy += size;
    pga += skip_a;
    pgb += skip_b;
  }
}

}  // namespace devices
}  // namespace primitiv
