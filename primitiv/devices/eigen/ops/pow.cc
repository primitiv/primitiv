#include <primitiv/config.h>

#include <cmath>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X_CONST(pow_const_r, x.pow(k));
EIGEN_DEV_BW_X_CONST(pow_const_r, k * gy * y / x);

EIGEN_DEV_FW_X_CONST(pow_const_l, ::Eigen::pow(k, x));
EIGEN_DEV_BW_X_CONST(pow_const_l, std::log(k) * gy * y);

EIGEN_DEV_FW_X_SCALAR(pow_scalar_r, x.pow(k));

EIGEN_DEV_FW_X_SCALAR(pow_scalar_l, ::Eigen::pow(k, x));

EIGEN_DEV_FW_AB(pow, a.pow(b));

void Eigen::pow_bw_impl(
    const Tensor &a_, const Tensor &b_, const Tensor &y_, const Tensor &gy_,
    Tensor &ga_, Tensor &gb_) {
  const std::uint32_t size = gy_.shape().volume();
  const std::uint32_t bs = gy_.shape().batch();
  const std::uint32_t skip_a = ga_.shape().has_batch() * size;
  const std::uint32_t skip_b = gb_.shape().has_batch() * size;
  const float *pa = CDATA(a_);
  const float *pb = CDATA(b_);
  const float *py = CDATA(y_);
  const float *pgy = CDATA(gy_);
  float *pga = MDATA(ga_);
  float *pgb = MDATA(gb_);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    EMap<const EArrayXf> a(pa, size);
    EMap<const EArrayXf> b(pb, size);
    EMap<const EArrayXf> y(py, size);
    EMap<const EArrayXf> gy(pgy, size);
    EMap<EArrayXf>(pga, size) += gy * y * b / a;
    EMap<EArrayXf>(pgb, size) += gy * y * a.log();
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
