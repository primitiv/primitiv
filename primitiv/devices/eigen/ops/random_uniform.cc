#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::random_uniform_impl(float lower, float upper, Tensor &y) {
  randomizer_.fill_uniform(lower, upper, y.shape().size(), MDATA(y));
}

}  // namespace devices
}  // namespace primitiv
