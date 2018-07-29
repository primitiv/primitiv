#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::inplace_multiply_const_impl(float k, Tensor &x) {
  EMap<EArrayXf>(MDATA(x), x.shape().size()) *= k;
}

}  // namespace devices
}  // namespace primitiv
