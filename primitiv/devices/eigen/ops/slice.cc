#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::slice_fw_impl(
    const Tensor &x, std::uint32_t dim, std::uint32_t offset, Tensor &y) {
  // TODO(odashi): Optimize this functions using Eigen operations.

  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t span = base * y.shape()[dim];
  const std::uint32_t skip = base * x.shape()[dim];
  const std::uint32_t repeat = y.shape().size() / span;

  float *dest = MDATA(y);
  const float *src = CDATA(x) + base * offset;
  for (std::uint32_t i = 0; i < repeat; ++i) {
    const float *sp = src;
    REPEAT_OP(j, span, *dest++ = *sp++);
    src += skip;
  }
}

void Eigen::slice_bw_impl(
    const Tensor &gy, std::uint32_t dim, std::uint32_t offset, Tensor &gx) {
  // TODO(odashi): Optimize this functions using Eigen operations.

  const Shape &sy = gy.shape();
  const Shape &sx = gx.shape();
  const std::uint32_t base = sx.lower_volume(dim);
  const std::uint32_t span = base * sy[dim];
  const std::uint32_t skip = base * sx[dim];
  const std::uint32_t repeat = sx.volume() / skip;
  const std::uint32_t bs = std::max(sx.batch(), sy.batch());
  const std::uint32_t b_skip_d = sx.has_batch() * sx.volume();
  const std::uint32_t b_skip_s = sy.has_batch() * sy.volume();
  float *dest = MDATA(gx) + base * offset;
  const float *src = CDATA(gy);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    float *dp = dest;
    const float *sp = src;
    for (std::uint32_t i = 0; i < repeat; ++i) {
      float *ddp = dp;
      REPEAT_OP(j, span, *ddp++ += *sp++);
      dp += skip;
    }
    dest += b_skip_d;
    src += b_skip_s;
  }
}

}  // namespace devices
}  // namespace primitiv
