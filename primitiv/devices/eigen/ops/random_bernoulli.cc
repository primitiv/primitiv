#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::random_bernoulli_impl(float p, Tensor &y) {
  randomizer_.fill_bernoulli(p, y.shape().size(), MDATA(y));
}

}  // namespace devices
}  // namespace primitiv
