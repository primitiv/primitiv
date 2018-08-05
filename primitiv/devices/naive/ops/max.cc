#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::max_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t repeat = y.shape().size();
  const std::uint32_t skip1 = y.shape().lower_volume(dim);
  const std::uint32_t skip2 = skip1 * n;
  const float *px = CDATA(x);
  float *py = MDATA(y);
  for (std::uint32_t i = 0; i < repeat; ++i) {
    std::uint32_t offset = i % skip1 + (i / skip1) * skip2;
    float tmp = px[offset];
    for (std::uint32_t j = 0; j < n; ++j) {
      if (px[offset] > tmp) {
        tmp = px[offset];
      }
      offset += skip1;
    }
    py[i] = tmp;
  }
}

void Naive::max_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, std::uint32_t dim, Tensor &gx) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t repeat = y.shape().size();
  const std::uint32_t skip1 = y.shape().lower_volume(dim);
  const std::uint32_t skip2 = skip1 * n;
  const float *py = CDATA(y);
  const float *px = CDATA(x);
  const float *pgy = CDATA(gy);
  float *pgx = MDATA(gx);
  for (std::uint32_t i = 0; i < repeat; ++i) {
    const float maxval = py[i];
    std::uint32_t offset = i % skip1 + (i / skip1) * skip2;
    for (std::uint32_t j = 0; j < n; ++j) {
      if (px[offset] == maxval) {
        pgx[offset] += pgy[i];
        break;
      }
      offset += skip1;
    }
  }
}

}  // namespace devices
}  // namespace primitiv
