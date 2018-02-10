#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

__global__ void inplace_add_dev(
    const half *px,
    std::uint32_t volume, std::uint32_t mbx, std::uint32_t mby,
    half *py) {
  const std::uint32_t i = IDX;

  if (i < volume) {
    const std::uint32_t offset = blockIdx.y * volume;
    const float x = ::__half2float(px[i + mbx * offset]);
    const std::uint32_t iy = i + mby * offset;
    const std::uint32_t shift = 16 * (iy & 1);
    const std::uint32_t filter = 0xffff << (16 - shift);

    std::uint32_t *addr = reinterpret_cast<std::uint32_t *>(py) + (iy >> 1);
    std::uint32_t oldval = *addr;
    std::uint32_t assumed;

    do {
      assumed = oldval;
      const half a = ::__ushort_as_half((oldval >> shift) & 0xffff);
      const half b = ::__float2half(::__half2float(a) + x);
      const std::uint32_t newval
        = (oldval & filter) | (::__half_as_ushort(b) << shift);
      oldval = ::atomicCAS(addr, assumed, newval);
    } while (oldval != assumed);
  }
}

} // namespace

namespace primitiv {
namespace devices {

void CUDA16::inplace_add_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t volume = y.shape().volume();
  const std::uint32_t g1 = GRID_SIZE(volume, dim1_x_);
  const std::uint32_t bs = std::max(x.shape().batch(), y.shape().batch());

  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::inplace_add_dev<<<dim3(g1, bs, 1), dim1_x_>>>(
      CDATA(half, x),
      volume, x.shape().has_batch(), y.shape().has_batch(),
      MDATA(half, y));
}

}  // namespace devices
}  // namespace primitiv
