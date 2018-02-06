#include <primitiv/config.h>

#include <primitiv/naive_device.h>
#include <primitiv/device_ops/naive/common.h>

namespace primitiv {
namespace devices {

void Naive::reset_tensor_impl(float k, Tensor &x) {
  float *dest = MDATA(x);
  const std::uint32_t size = x.shape().size();
  REPEAT_OP(i, size, dest[i] = k);
}

}  // namespace devices
}  // namespace primitiv
