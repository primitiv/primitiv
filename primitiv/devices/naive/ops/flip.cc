#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::flip_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const Shape &s = x.shape();
  const std::uint32_t n = s[dim];
  const std::uint32_t skip = s.lower_volume(dim);
  const std::uint32_t r = s.size() / n;
  const float *px = CDATA(x);
  float *py = MDATA(y);
  for (std::uint32_t j = 0; j < n; ++j) {
    for (std::uint32_t i = 0; i < r; ++i) {
      const std::uint32_t offset = i * n - i % skip * (n - 1);
      py[offset + j * skip] = px[offset + (n - j - 1) * skip];
    }
  }
}

void Naive::flip_bw_impl(const Tensor &gy, std::uint32_t dim, Tensor &gx) {
  const Shape &s = gx.shape();
  const std::uint32_t n = s[dim];
  const std::uint32_t skip = s.lower_volume(dim);
  const std::uint32_t r = s.size() / n;
  const float *pgy = CDATA(gy);
  float *pgx = MDATA(gx);
  for (std::uint32_t j = 0; j < n; ++j) {
    for (std::uint32_t i = 0; i < r; ++i) {
      const std::uint32_t offset = i * n - i % skip * (n - 1);
      pgx[offset + j * skip] += pgy[offset + (n - j - 1) * skip];
    }
  }
}

}  // namespace devices
}  // namespace primitiv

