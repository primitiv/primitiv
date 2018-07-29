#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X_CONST(add_const, x + k);
EIGEN_DEV_BW_X_CONST(add_const, gy);

EIGEN_DEV_FW_X_SCALAR(add_scalar, x + k);

EIGEN_DEV_FW_AB(add, a + b);

void Eigen::add_bw_impl(
    const Tensor &, const Tensor &, const Tensor &, const Tensor &gy_,
    Tensor &ga_, Tensor &gb_) {
  const std::uint32_t size = gy_.shape().volume();
  const std::uint32_t bs = gy_.shape().batch();
  const std::uint32_t skip_a = ga_.shape().has_batch() * size;
  const std::uint32_t skip_b = gb_.shape().has_batch() * size;
  const float *pgy = CDATA(gy_);
  float *pga = MDATA(ga_);
  float *pgb = MDATA(gb_);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    EMap<const EArrayXf> gy(pgy, size);
    EMap<EArrayXf>(pga, size) += gy;
    EMap<EArrayXf>(pgb, size) += gy;
    pgy += size;
    pga += skip_a;
    pgb += skip_b;
  }
}

}  // namespace devices
}  // namespace primitiv
