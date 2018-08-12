#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::copy_tensor_impl(const Tensor &x, Tensor &y) {
  switch (x.device().type()) {
    case DeviceType::NAIVE:
      reset_tensor_by_array(CDATA(x), y);
      break;
    default:
      reset_tensor_by_vector(x.to_vector(), y);
  }
}

}  // namespace devices
}  // namespace primitiv
