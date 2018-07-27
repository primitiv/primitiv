#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::pick_fw_impl(
    const Tensor &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim,
    Tensor &y) {
  // TODO(odashi): Optimize this functions using Eigen operations.

  const std::uint32_t bs = y.shape().batch();
  const std::uint32_t skip_x = x.shape().has_batch() * x.shape().volume();
  const std::uint32_t skip_i = ids.size() > 1;
  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t skip = base * x.shape()[dim];
  const std::uint32_t repeat = y.shape().volume() / base;

  float *dest = MDATA(y);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    const float *src = CDATA(x) + batch * skip_x + base * ids[batch * skip_i];
    for (std::uint32_t i = 0; i < repeat; ++i) {
      const float *sp = src;
      REPEAT_OP(j, base, *dest++ = *sp++);
      src += skip;
    }
  }
}

void Eigen::pick_bw_impl(
    const Tensor &gy, const std::vector<std::uint32_t>& ids, std::uint32_t dim,
    Tensor &gx) {
  // TODO(odashi): Optimize this functions using Eigen operations.

  const std::uint32_t bs = gy.shape().batch();
  const std::uint32_t skip_x = gx.shape().has_batch() * gx.shape().volume();
  const std::uint32_t skip_i = ids.size() > 1;
  const std::uint32_t base = gy.shape().lower_volume(dim);
  const std::uint32_t skip = base * gx.shape()[dim];
  const std::uint32_t repeat = gy.shape().volume() / base;
  const float *src = CDATA(gy);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    float *dest = MDATA(gx) + batch * skip_x + base * ids[batch * skip_i];
    for (std::uint32_t i = 0; i < repeat; ++i) {
      float *dp = dest;
      REPEAT_OP(j, base, *dp++ += *src++);
      dest += skip;
    }
  }
}

}  // namespace devices
}  // namespace primitiv
