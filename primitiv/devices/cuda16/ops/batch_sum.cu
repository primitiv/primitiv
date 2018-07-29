#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void batch_sum_fw_dev(
    const half *px, std::uint32_t size, std::uint32_t batch, half *py) {
  const std::uint32_t i = IDX;
  if (i < size) {
    float temp = .0f;
    px += i;
    for (std::uint32_t j = 0; j < batch; ++j, px += size) {
      temp += ::__half2float(*px);
    }
    py[i] = ::__float2half(temp);
  }
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::batch_sum_fw_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::batch_sum_fw_dev<<<g1, dim1_x_>>>(
      CDATA(half, x), size, x.shape().batch(), MDATA(half, y));
}

}  // namespace devices
}  // namespace primitiv
