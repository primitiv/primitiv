#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::reset_tensor_impl(float k, Tensor &x) {
  EMap<EArrayXf>(MDATA(x), x.shape().size()).setConstant(k);
}

}  // namespace devices
}  // namespace primitiv
