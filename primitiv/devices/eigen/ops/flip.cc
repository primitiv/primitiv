#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::flip_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  if (dim == 0) {
    const std::uint32_t di = x.shape()[0];
    const std::uint32_t dj = x.shape().volume() / di;
    const float *src = CDATA(x);
    float *dest = MDATA(y);
    const std::uint32_t skip = x.shape().volume();
    const std::uint32_t bs = x.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      EMap<const EMatrixXf> xx(src + n * skip, di, dj);
      EMap<EMatrixXf> yy(dest + n * skip, di, dj);
      yy.noalias() = xx.colwise().reverse();
    }
  } else if (dim == x.shape().depth() - 1) {
    const std::uint32_t di = x.shape().lower_volume(dim);
    const std::uint32_t dj = x.shape()[dim];
    const float *src = CDATA(x);
    float *dest = MDATA(y);
    const std::uint32_t skip = x.shape().volume();
    const std::uint32_t bs = x.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      EMap<const EMatrixXf> xx(src + n * skip, di, dj);
      EMap<EMatrixXf> yy(dest + n * skip, di, dj);
      yy.noalias() = xx.rowwise().reverse();
    }
  } else {
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
}

void Eigen::flip_bw_impl(const Tensor &gy, std::uint32_t dim, Tensor &gx) {
  if (dim == 0) {
    const std::uint32_t di = gx.shape()[0];
    const std::uint32_t dj = gx.shape().volume() / di;
    const float *src = CDATA(gy);
    float *dest = MDATA(gx);
    const std::uint32_t skip = gx.shape().volume();
    const std::uint32_t bs = gx.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      EMap<const EMatrixXf> gyy(src + n * skip, di, dj);
      EMap<EMatrixXf> gxx(dest + n * skip, di, dj);
      gxx.noalias() += gyy.colwise().reverse();
    }
  } else if (dim == gx.shape().depth() - 1) {
    const std::uint32_t di = gx.shape().lower_volume(dim);
    const std::uint32_t dj = gx.shape()[dim];
    const float *src = CDATA(gy);
    float *dest = MDATA(gx);
    const std::uint32_t skip = gx.shape().volume();
    const std::uint32_t bs = gx.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      EMap<const EMatrixXf> gyy(src + n * skip, di, dj);
      EMap<EMatrixXf> gxx(dest + n * skip, di, dj);
      gxx.noalias() += gyy.rowwise().reverse();
    }
  } else {
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
}

}  // namespace devices
}  // namespace primitiv

