#include <primitiv/config.h>

#include <primitiv/eigen_device.h>
#include <primitiv/device_ops/common_eigen.h>

namespace primitiv {
namespace devices {

void Eigen::identity_impl(Tensor &y) {
  const std::size_t size = y.shape()[0];
  EMap<EMatrixXf>(MDATA(y), size, size).setIdentity();
}

}  // namespace devices
}  // namespace primitiv
