#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::reset_tensor_by_array_impl(const float values[], Tensor &x) {
  const std::size_t size = x.shape().size();
  EMap<EArrayXf>(MDATA(x), size) = EMap<const EArrayXf>(values, size);
}

}  // namespace devices
}  // namespace primitiv
