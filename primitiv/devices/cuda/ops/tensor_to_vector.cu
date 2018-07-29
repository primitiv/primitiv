#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace primitiv {
namespace devices {

std::vector<float> CUDA::tensor_to_vector_impl(const Tensor &x) {
  const std::uint32_t size = x.shape().size();
  std::vector<float> ret(size);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        ret.data(), CDATA(x), sizeof(float) * size, cudaMemcpyDeviceToHost));
  return ret;
}

}  // namespace devices
}  // namespace primitiv
