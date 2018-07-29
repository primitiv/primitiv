#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::batch_slice_fw_impl(
    const Tensor &x, std::uint32_t offset, Tensor &y) {
  // TODO(chantera): Optimize this functions using Eigen operations.

  const std::uint32_t volume = y.shape().volume();
  const std::uint32_t repeat = y.shape().batch();

  float *dest = MDATA(y);
  const float *src = CDATA(x) + volume * offset;
  std::copy(src, src + volume * repeat, dest);
}

void Eigen::batch_slice_bw_impl(
    const Tensor &gy, std::uint32_t offset, Tensor &gx) {
  // TODO(chantera): Optimize this functions using Eigen operations.

  const std::uint32_t volume = gy.shape().volume();
  const std::uint32_t repeat = gy.shape().batch();

  float *dest = MDATA(gx) + volume * offset;
  const float *src = CDATA(gy);
  REPEAT_OP(i, volume * repeat, *dest++ += *src++);
}

}  // namespace devices
}  // namespace primitiv
