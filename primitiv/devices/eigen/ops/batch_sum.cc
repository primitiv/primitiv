#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::batch_sum_fw_impl(const Tensor &x_, Tensor &y_) {
  const std::size_t size = x_.shape().volume();
  const std::size_t bs = x_.shape().batch();

  const float *px = CDATA(x_);
  EMap<EArrayXf> y(MDATA(y_), size);
  y = EMap<const EArrayXf>(px, size);
  px += size;

  for (std::size_t i = 1; i < bs; ++i) {
    y += EMap<const EArrayXf>(px, size);
    px += size;
  }
}

}  // namespace devices
}  // namespace primitiv
