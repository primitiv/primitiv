#include <primitiv/config.h>

#include <primitiv/core/naive_device.h>
#include <primitiv/device_ops/naive/common.h>

namespace primitiv {
namespace devices {

void Naive::inplace_multiply_const_impl(float k, Tensor &x) {
  const std::uint32_t size = x.shape().size();
  float *dest = MDATA(x);
  REPEAT_OP(i, size, dest[i] *= k);
}

}  // namespace devices
}  // namespace primitiv
