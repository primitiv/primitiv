#include <primitiv/config.h>

#include <cmath>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::logsumexp_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  // TODO(odashi): Optimize this functions using Eigen operations.

  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t repeat = y.shape().size();
  const std::uint32_t skip1 = y.shape().lower_volume(dim);
  const std::uint32_t skip2 = skip1 * n;
  float *dest = MDATA(y);
  const float *src = CDATA(x);
  for (std::uint32_t i = 0; i < repeat; ++i) {
    // TODO(odashi): This calculation might generate large errors.
    std::uint32_t offset = i % skip1 + (i / skip1) * skip2;
    float tmp = src[offset];
    for (std::uint32_t j = 1; j < n; ++j) {
      offset += skip1;
      float arg = src[offset];
      tmp = tmp > arg
        ? tmp + std::log(1. + std::exp(arg - tmp))
        : arg + std::log(1. + std::exp(tmp - arg));
    }
    dest[i] = tmp;
  }
}

}  // namespace devices
}  // namespace primitiv
