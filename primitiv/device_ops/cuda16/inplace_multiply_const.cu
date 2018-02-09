#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

__global__ void inplace_multiply_const_dev(
    float k, std::uint32_t size, half *px) {
  const std::uint32_t i = IDX;
  if (i < size) px[i] = __float2half(__half2float(px[i]) * k);
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::inplace_multiply_const_impl(float k, Tensor &x) {
  const std::uint32_t size = x.shape().size();
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::inplace_multiply_const_dev<<<g1, dim1_x_>>>(k, size, MDATA(half, x));
}

}  // namespace devices
}  // namespace primitiv
