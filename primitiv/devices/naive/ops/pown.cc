#include <primitiv/config.h>

#include <cmath>
#include <limits>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::pown_fw_impl(const Tensor &x, std::int32_t k, Tensor &y) {
  float *dest = MDATA(y);
  const float *src = CDATA(x);
  const std::uint32_t size = x.shape().size();

  // NOTE(odashi):
  // std::abs(-0x80000000) is UB under 2's complement systems.
  // However, this value should be also evaluated as 0x80000000 by directly
  // casting to std::uint32_t.
  const std::int32_t min_k = std::numeric_limits<std::int32_t>::min();
  const std::uint32_t abs_k = (k == min_k) ? min_k : std::abs(k);

  for (std::uint32_t i = 0; i < size; ++i) {
    // Performs the exponentation-by-squaring method.
    float ret = 1.;
    float factor = src[i];
    std::uint32_t remain = abs_k;
    while (remain) {
      if (remain & 1) ret *= factor;
      factor *= factor;
      remain >>= 1;
    }
    dest[i] = k >= 0 ? ret : 1. / ret;
  }
}

void Naive::pown_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, std::int32_t k,
    Tensor &gx) {
  const float *px = CDATA(x);
  const float *py = CDATA(y);
  const float *pgy = CDATA(gy);
  float *pgx = MDATA(gx);
  const std::uint32_t size = x.shape().size();
  REPEAT_OP(i, size, pgx[i] += k * pgy[i] * py[i] / px[i]);
}

}  // namespace devices
}  // namespace primitiv
