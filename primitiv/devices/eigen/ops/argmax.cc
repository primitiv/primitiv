#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

std::vector<std::uint32_t> Eigen::argmax_impl(
    const Tensor &x, std::uint32_t dim) {
  // TODO(odashi): Optimize this functions using Eigen operations.

  const Shape &s = x.shape();
  const std::uint32_t n = s[dim];
  const std::uint32_t repeat = s.size() / n;
  const std::uint32_t skip1 = s.lower_volume(dim);
  const std::uint32_t skip2 = skip1 * n;
  const float *src = CDATA(x);
  std::vector<std::uint32_t> ret;
  ret.reserve(repeat);
  for (std::uint32_t i = 0; i < repeat; ++i) {
    std::uint32_t offset = i % skip1 + (i / skip1) * skip2;
    float max_val = src[offset];
    std::uint32_t argmax_val = 0;
    for (std::uint32_t j = 1; j < n; ++j) {
      offset += skip1;
      if (src[offset] > max_val) {
        max_val = src[offset];
        argmax_val = j;
      }
    }
    ret.emplace_back(argmax_val);
  }
  return ret;
}

}  // namespace devices
}  // namespace primitiv
