#include <primitiv/config.h>

#include <primitiv/core/naive_device.h>
#include <primitiv/device_ops/naive/common.h>

namespace primitiv {
namespace devices {

void Naive::identity_impl(Tensor &y) {
  reset_tensor_impl(0, y);
  float *dest = MDATA(y);
  const std::uint32_t size = y.shape()[0];
  REPEAT_OP(i, size, dest[i * (size + 1)] = 1);
}

}  // namespace devices
}  // namespace primitiv
