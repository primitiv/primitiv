#include <primitiv/config.h>

#include <primitiv/eigen_device.h>
#include <primitiv/device_ops/eigen/common.h>

namespace primitiv {
namespace devices {

void Eigen::random_log_normal_impl(float mean, float sd, Tensor &y) {
  randomizer_.fill_log_normal(mean, sd, y.shape().size(), MDATA(y));
}

}  // namespace devices
}  // namespace primitiv
