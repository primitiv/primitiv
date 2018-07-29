#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::matmul_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
  const std::uint32_t d1 = a.shape()[0];
  const std::uint32_t d2 = a.shape()[1];
  const std::uint32_t d3 = b.shape()[1];
  const std::uint32_t bs = y.shape().batch();
  const std::uint32_t dest_shift = d1 * d3;
  const std::uint32_t src_a_shift = a.shape().has_batch() * d1 * d2;
  const std::uint32_t src_b_shift = b.shape().has_batch() * d2 * d3;

  float *dest = MDATA(y);
  const float *src_a = CDATA(a);
  const float *src_b = CDATA(b);

  for (std::uint32_t batch = 0; batch < bs; ++batch) {
    for (std::uint32_t n = 0; n < dest_shift; ++n) {
      dest[n] = 0;
    }
    for (std::uint32_t k = 0; k < d3; k += 8) {
      const std::uint32_t ek = std::min(k + 8, d3);
      for (std::uint32_t i = 0; i < d1; i += 8) {
        const std::uint32_t ei = std::min(i + 8, d1);
        for (std::uint32_t j = 0; j < d2; j += 8) {
          const std::uint32_t ej = std::min(j + 8, d2);
          for (std::uint32_t kk = k; kk < ek; ++kk) {
            const std::uint32_t kk_d1 = kk * d1;
            const std::uint32_t kk_d2 = kk * d2;
            for (std::uint32_t ii = i; ii < ei; ++ii) {
              float tmp = 0;
              for (std::uint32_t jj = j; jj < ej; ++jj) {
                tmp += src_a[ii + jj * d1] * src_b[jj + kk_d2];
              }
              dest[ii + kk_d1] += tmp;
            }
          }
        }
      }
    }
    dest += dest_shift;
    src_a += src_a_shift;
    src_b += src_b_shift;
  }
}

void Naive::matmul_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  // TODO(odashi): This code could be slow and requires memory. Fix this.
  inplace_add_impl(matmul_fw(gy, transpose_fw(b)), ga);
  inplace_add_impl(matmul_fw(transpose_fw(a), gy), gb);
}

}  // namespace devices
}  // namespace primitiv
