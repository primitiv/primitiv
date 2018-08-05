#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::batch_sum_fw_impl(const Tensor &x, Tensor &y) {
  float *dest = MDATA(y);
  const float *src = CDATA(x);
  const std::uint32_t bs = x.shape().batch();
  const std::uint32_t size = y.shape().size();
  for (std::uint32_t i = 0; i < size; ++i) {
    float temp = 0;
    for (std::uint32_t batch = 0, pos = i; batch < bs; ++batch, pos += size) {
      temp += src[pos];
    }
    dest[i] = temp;
  }
}

}  // namespace devices
}  // namespace primitiv
