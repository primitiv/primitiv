#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::permute_dims_fw_impl(
    const Tensor &x, const std::vector<std::uint32_t> &perm, Tensor &y) {
  const std::uint32_t volume = x.shape().volume();
  const std::uint32_t bs = x.shape().batch();
  float *dest = MDATA(y);
  const float *src = CDATA(x);
  std::vector<std::uint32_t> x_strides(perm.size());
  std::vector<std::uint32_t> y_strides(perm.size());
  for (std::uint32_t i = 0; i < perm.size(); ++i) {
    x_strides[perm.size() - i - 1] = x.shape().lower_volume(i);
    y_strides[perm.size() - perm[i] - 1] = y.shape().lower_volume(i);
  }
  for (std::uint32_t k = 0; k < bs; ++k) {
    for (std::uint32_t i = 0; i < volume; ++i) {
      std::uint32_t tmp = i;
      std::uint32_t j = 0;
      for (std::uint32_t d = 0; d < perm.size(); ++d) {
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
  float *pgx = MDATA(gx);
  const float *pgy = CDATA(gy);
  std::vector<std::uint32_t> x_strides(perm.size());
  std::vector<std::uint32_t> y_strides(perm.size());
  for (std::uint32_t i = 0; i < perm.size(); ++i) {
    x_strides[perm.size() - i - 1] = gx.shape().lower_volume(i);
    y_strides[perm.size() - perm[i] - 1] = gy.shape().lower_volume(i);
  }
  for (std::uint32_t k = 0; k < bs; ++k) {
    for (std::uint32_t i = 0; i < volume; ++i) {
      std::uint32_t tmp = i;
      std::uint32_t j = 0;
      for (std::uint32_t d = 0; d < perm.size(); ++d) {
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
