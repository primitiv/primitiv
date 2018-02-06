#include <primitiv/config.h>

#include <primitiv/cuda_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/common_cuda.h>

namespace primitiv {
namespace devices {

std::vector<float> CUDA::tensor_to_vector_impl(const Tensor &x) {
  const std::uint32_t size = x.shape().size();
  std::vector<float> ret(size);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        &ret[0], CDATA(x), sizeof(float) * size, cudaMemcpyDeviceToHost));
  return ret;
}

}  // namespace devices
}  // namespace primitiv
