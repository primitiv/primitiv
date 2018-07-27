#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::transpose_fw_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t di = x.shape()[0];
  const std::uint32_t dj = x.shape()[1];
  const std::uint32_t ms = di * dj;
  const std::uint32_t bs = x.shape().batch();

  const float *src = CDATA(x);
  float *dest = MDATA(y);

  for (std::uint32_t n = 0; n < bs; ++n) {
    EMap<const EMatrixXf> xx(src + n * ms, di, dj);
    EMap<EMatrixXf> yy(dest + n * ms, dj, di);
    yy.noalias() = xx.transpose();
  }
}

void Eigen::transpose_bw_impl(
    const Tensor &, const Tensor &, const Tensor &gy, Tensor &gx) {
  const std::uint32_t di = gx.shape()[0];
  const std::uint32_t dj = gx.shape()[1];
  const std::uint32_t ms = di * dj;
  const std::uint32_t bs = gx.shape().batch();

  const float *src = CDATA(gy);
  float *dest = MDATA(gx);

  for (std::uint32_t n = 0; n < bs; ++n) {
    EMap<const EMatrixXf> gyy(src + n * ms, dj, di);
    EMap<EMatrixXf> gxx(dest + n * ms, di, dj);
    gxx.noalias() += gyy.transpose();
  }
}

}  // namespace devices
}  // namespace primitiv
