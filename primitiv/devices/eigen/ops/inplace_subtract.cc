#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::inplace_subtract_impl(const Tensor &x, Tensor &y) {
  const Shape &sx = x.shape();
  const Shape &sy = y.shape();
  const std::uint32_t size = sy.volume();
  const std::uint32_t bs = std::max(sx.batch(), sy.batch());
  const std::uint32_t skip_y = sy.has_batch() * size;
  const std::uint32_t skip_x = sx.has_batch() * size;
  float *py = MDATA(y);
  const float *px = CDATA(x);
  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    EMap<EArrayXf>(py, size) -= EMap<const EArrayXf>(px, size);
    py += skip_y;
    px += skip_x;
  }
}

}  // namespace devices
}  // namespace primitiv
