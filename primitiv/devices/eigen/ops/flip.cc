#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::flip_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const float *px = CDATA(x);
  float *py = MDATA(y);
  if (dim == 0) {
    const std::uint32_t di = x.shape()[0];
    const std::uint32_t dj = x.shape().volume() / di;
    const std::uint32_t skip = x.shape().volume();
    const std::uint32_t bs = x.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      EMap<const EMatrixXf> xx(px + n * skip, di, dj);
      EMap<EMatrixXf> yy(py + n * skip, di, dj);
      yy.noalias() = xx.colwise().reverse();
    }
  } else if (dim == x.shape().depth() - 1) {
    const std::uint32_t dj = x.shape()[dim];
    const std::uint32_t di = x.shape().volume() / dj;
    const std::uint32_t skip = x.shape().volume();
    const std::uint32_t bs = x.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      EMap<const EMatrixXf> xx(px + n * skip, di, dj);
      EMap<EMatrixXf> yy(py + n * skip, di, dj);
      yy.noalias() = xx.rowwise().reverse();
    }
  } else {
    const std::uint32_t n = x.shape()[dim];
    const std::uint32_t skip = x.shape().lower_volume(dim);
    const std::uint32_t r = x.shape().size() / n;
    for (std::uint32_t j = 0; j < n; ++j) {
      for (std::uint32_t i = 0; i < r; ++i) {
        const std::uint32_t offset = i * n - i % skip * (n - 1);
        py[offset + j * skip] = px[offset + (n - j - 1) * skip];
      }
    }
  }
}

void Eigen::flip_bw_impl(const Tensor &gy, std::uint32_t dim, Tensor &gx) {
  const float *pgy = CDATA(gy);
  float *pgx = MDATA(gx);
  if (dim == 0) {
    const std::uint32_t di = gx.shape()[0];
    const std::uint32_t dj = gx.shape().volume() / di;
    const std::uint32_t skip = gx.shape().volume();
    const std::uint32_t bs = gx.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      EMap<const EMatrixXf> gyy(pgy + n * skip, di, dj);
      EMap<EMatrixXf> gxx(pgx + n * skip, di, dj);
      gxx.noalias() += gyy.colwise().reverse();
    }
  } else if (dim == gx.shape().depth() - 1) {
    const std::uint32_t dj = gx.shape()[dim];
    const std::uint32_t di = gx.shape().volume() / dj;
    const std::uint32_t skip = gx.shape().volume();
    const std::uint32_t bs = gx.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      EMap<const EMatrixXf> gyy(pgy + n * skip, di, dj);
      EMap<EMatrixXf> gxx(pgx + n * skip, di, dj);
      gxx.noalias() += gyy.rowwise().reverse();
    }
  } else {
    const std::uint32_t n = gx.shape()[dim];
    const std::uint32_t skip = gx.shape().lower_volume(dim);
    const std::uint32_t r = gx.shape().size() / n;
    for (std::uint32_t j = 0; j < n; ++j) {
      for (std::uint32_t i = 0; i < r; ++i) {
        const std::uint32_t offset = i * n - i % skip * (n - 1);
        pgx[offset + j * skip] += pgy[offset + (n - j - 1) * skip];
      }
    }
  }
}

}  // namespace devices
}  // namespace primitiv

