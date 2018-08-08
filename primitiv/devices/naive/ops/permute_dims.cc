#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::permute_dims_fw_impl(
    const Tensor &x, const std::vector<std::uint32_t> &perm, Tensor &y) {
  const std::uint32_t volume = x.shape().volume();
  const std::uint32_t bs = x.shape().batch();
  const std::uint32_t ndims = perm.size();
  float *dest = MDATA(y);
  const float *src = CDATA(x);
  std::vector<std::uint32_t> x_strides(ndims);
  std::vector<std::uint32_t> y_strides(ndims);
  std::uint32_t x_stride_tmp = 1;
  std::uint32_t y_stride_tmp = 1;
  for (std::uint32_t i = 0; i < ndims; ++i) {
    x_strides[ndims - i - 1] = x_stride_tmp;
    y_strides[ndims - perm[i] - 1] = y_stride_tmp;
    x_stride_tmp *= x.shape()[i];
    y_stride_tmp *= y.shape()[i];
  }
  for (std::uint32_t k = 0; k < bs; ++k) {
    for (std::uint32_t i = 0; i < volume; ++i) {
      std::uint32_t tmp = i;
      std::uint32_t j = 0;
      for (std::uint32_t d = 0; d < ndims; ++d) {
        const std::uint32_t p = tmp / x_strides[d];
        tmp -= p * x_strides[d];
        j += p * y_strides[d];
      }
      dest[j] = src[i];
    }
    src += volume;
    dest += volume;
  }
}

void Naive::permute_dims_bw_impl(
    const Tensor &, const Tensor &, const Tensor &gy,
    const std::vector<std::uint32_t> &perm, Tensor &gx) {
  const std::uint32_t volume = gx.shape().volume();
  const std::uint32_t bs = gx.shape().batch();
  const std::uint32_t ndims = perm.size();
  float *pgx = MDATA(gx);
  const float *pgy = CDATA(gy);
  std::vector<std::uint32_t> x_strides(ndims);
  std::vector<std::uint32_t> y_strides(ndims);
  std::uint32_t x_stride_tmp = 1;
  std::uint32_t y_stride_tmp = 1;
  for (std::uint32_t i = 0; i < ndims; ++i) {
    x_strides[ndims - i - 1] = x_stride_tmp;
    y_strides[ndims - perm[i] - 1] = y_stride_tmp;
    x_stride_tmp *= gx.shape()[i];
    y_stride_tmp *= gy.shape()[i];
  }
  for (std::uint32_t k = 0; k < bs; ++k) {
    for (std::uint32_t i = 0; i < volume; ++i) {
      std::uint32_t tmp = i;
      std::uint32_t j = 0;
      for (std::uint32_t d = 0; d < ndims; ++d) {
        const std::uint32_t p = tmp / x_strides[d];
        tmp -= p * x_strides[d];
        j += p * y_strides[d];
      }
      pgx[i] += pgy[j];
    }
    pgx += volume;
    pgy += volume;
  }
}

}  // namespace devices
}  // namespace primitiv
