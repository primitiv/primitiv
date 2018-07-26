#include <primitiv/config.h>

#include <primitiv/core/naive_device.h>
#include <primitiv/device_ops/naive/common.h>

namespace primitiv {
namespace devices {

void Naive::batch_concat_fw_impl(
    const std::vector<const Tensor *> &xs, Tensor &y) {
  std::uint32_t offset = 0;
  for (const Tensor *x : xs) {
    const std::uint32_t span = x->shape().size();
    float *dest = MDATA(y) + offset;
    const float *src = CDATA(*x);
    std::copy(src, src + span, dest);
    offset += span;
  }
}

}  // namespace devices
}  // namespace primitiv
