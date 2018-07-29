#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::identity_impl(Tensor &y) {
  const std::size_t size = y.shape()[0];
  EMap<EMatrixXf>(MDATA(y), size, size).setIdentity();
}

}  // namespace devices
}  // namespace primitiv
