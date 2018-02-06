#include <primitiv/config.h>

#include <primitiv/eigen_device.h>
#include <primitiv/device_ops/common_eigen.h>

namespace primitiv {
namespace devices {

void Eigen::reset_tensor_impl(float k, Tensor &x) {
  EMap<EArrayXf>(MDATA(x), x.shape().size()).setConstant(k);
}

}  // namespace devices
}  // namespace primitiv
