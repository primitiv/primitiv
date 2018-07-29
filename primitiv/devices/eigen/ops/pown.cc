#include <primitiv/config.h>

#include <cmath>
#include <limits>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::pown_fw_impl(const Tensor &x_, std::int32_t k, Tensor &y_) {
  const std::size_t size = x_.shape().size();
  EMap<const EArrayXf> x(CDATA(x_), size);
  EMap<EArrayXf> y(MDATA(y_), size);
  y.setConstant(1.);

  // NOTE(odashi):
  // std::abs(-0x80000000) is UB under 2's complement systems.
  // However, this value should be also evaluated as 0x80000000 by directly
  // casting to std::uint32_t.
  const std::int32_t min_k = std::numeric_limits<std::int32_t>::min();
  std::uint32_t remain = (k == min_k) ? min_k : std::abs(k);
  EArrayXf factor = x;

  // Performs the exponentation-by-squaring method.
  while (remain) {
    if (remain & 1) y *= factor;
    factor *= factor;
    remain >>= 1;
  }

  if (k < 0) y = 1. / y;
}

void Eigen::pown_bw_impl(
    const Tensor &x_, const Tensor &y_, const Tensor &gy_, std::int32_t k,
    Tensor &gx_) {
  const std::size_t size = x_.shape().size();
  EMap<const EArrayXf> x(CDATA(x_), size);
  EMap<const EArrayXf> y(CDATA(y_), size);
  EMap<const EArrayXf> gy(CDATA(gy_), size);
  EMap<EArrayXf>(MDATA(gx_), size) += k * gy * y / x;
}

}  // namespace devices
}  // namespace primitiv
