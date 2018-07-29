#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::inplace_add_impl(const Tensor &x, Tensor &y) {
  const Shape &sx = x.shape();
  const Shape &sy = y.shape();
  const std::uint32_t size = sy.volume();
  const std::uint32_t bs = std::max(sx.batch(), sy.batch());
  const std::uint32_t b_skip_d = sy.has_batch() * size;
  const std::uint32_t b_skip_s = sx.has_batch() * size;
  float *dest = MDATA(y);
  const float *src = CDATA(x);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    REPEAT_OP(i, size, dest[i] += src[i]);
    dest += b_skip_d;
    src += b_skip_s;
  }
}

}  // namespace devices
}  // namespace primitiv
