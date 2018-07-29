#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::random_log_normal_impl(float mean, float sd, Tensor &y) {
  randomizer_.fill_log_normal(mean, sd, y.shape().size(), MDATA(y));
}

}  // namespace devices
}  // namespace primitiv
