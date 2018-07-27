#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::sum_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  // TODO(odashi): Optimize this functions using Eigen operations.

  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t repeat = y.shape().size();
  const std::uint32_t skip1 = y.shape().lower_volume(dim);
  const std::uint32_t skip2 = skip1 * n;
  float *dest = MDATA(y);
  const float *src = CDATA(x);
  for (std::uint32_t i = 0; i < repeat; ++i) {
    std::uint32_t offset = i % skip1 + (i / skip1) * skip2;
    float tmp = 0;
    for (std::uint32_t j = 0; j < n; ++j) {
      tmp += src[offset];
      offset += skip1;
    }
    dest[i] = tmp;
  }
}

}  // namespace devices
}  // namespace primitiv
