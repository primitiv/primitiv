#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::inplace_multiply_const_impl(float k, Tensor &x) {
  const std::uint32_t size = x.shape().size();
  float *dest = MDATA(x);
  REPEAT_OP(i, size, dest[i] *= k);
}

}  // namespace devices
}  // namespace primitiv
