#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::concat_fw_impl(
    const std::vector<const Tensor *> &xs, std::uint32_t dim, Tensor &y) {
  const std::uint32_t new_bs = y.shape().batch();
  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t skip = base * y.shape()[dim];
  const std::uint32_t repeat = y.shape().volume() / skip;

  std::uint32_t offset = 0;
  for (const Tensor *x : xs) {
    const std::uint32_t src_dim = x->shape()[dim];
    const std::uint32_t span = base * src_dim;
    const std::uint32_t b_skip = x->shape().has_batch() * span * repeat;
    float *dest = MDATA(y) + offset;
    const float *src = CDATA(*x);
    for (std::uint32_t batch = 0; batch < new_bs; ++batch) {
      const float *sp = src;
      for (std::uint32_t i = 0; i < repeat; ++i) {
        float *dp = dest;
        REPEAT_OP(j, span, *dp++ = *sp++);
        dest += skip;
      }
      src += b_skip;
    }
    offset += span;
  }
}

}  // namespace devices
}  // namespace primitiv
