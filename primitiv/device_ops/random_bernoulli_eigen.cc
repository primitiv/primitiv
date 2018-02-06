#include <primitiv/config.h>

#include <primitiv/eigen_device.h>
#include <primitiv/device_ops/common_eigen.h>

namespace primitiv {
namespace devices {

void Eigen::random_bernoulli_impl(float p, Tensor &y) {
  randomizer_.fill_bernoulli(p, y.shape().size(), MDATA(y));
}

}  // namespace devices
}  // namespace primitiv
