#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/internal/cuda/utils.h>
#include <primitiv/devices/cuda/ops/common.h>

namespace primitiv {
namespace devices {

void CUDA::reset_tensor_by_array_impl(const float values[], Tensor &x) {
  const std::uint32_t size = x.shape().size();
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        MDATA(x), values, sizeof(float) * size, cudaMemcpyHostToDevice));
}

}  // namespace devices
}  // namespace primitiv
