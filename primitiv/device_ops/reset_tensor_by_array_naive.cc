#include <primitiv/config.h>

#include <cstring>

#include <primitiv/naive_device.h>
#include <primitiv/device_ops/naive_utils.h>

namespace primitiv {
namespace devices {

void Naive::reset_tensor_by_array_impl(const float values[], Tensor &x) {
  std::memcpy(MDATA(x), values, sizeof(float) * x.shape().size());
}

}  // namespace devices
}  // namespace primitiv
