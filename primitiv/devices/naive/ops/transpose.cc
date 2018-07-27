#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::transpose_fw_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t d1 = x.shape()[0];
  const std::uint32_t d2 = x.shape()[1];
  const std::uint32_t ms = d1 * d2;
  const std::uint32_t bs = y.shape().batch();
  float *dest = MDATA(y);
  const float *src = CDATA(x);

  for (std::uint32_t k = 0; k < bs; ++k) {
    float *pd = dest;
    for (std::uint32_t j = 0; j < d2; ++j) {
      float *ppd = pd;
      for (std::uint32_t i = 0; i < d1; ++i) {
        *ppd = *src++;
        ppd += d2;
      }
      ++pd;
    }
    dest += ms;
  }
}

void Naive::transpose_bw_impl(
    const Tensor &, const Tensor &, const Tensor &gy, Tensor &gx) {
  // TODO(odashi): This code could be slow and requires memory. Fix this.
  inplace_add_impl(transpose_fw(gy), gx);
}

}  // namespace devices
}  // namespace primitiv
