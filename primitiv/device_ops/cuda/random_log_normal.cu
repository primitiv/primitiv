#include <primitiv/config.h>

#include <primitiv/core/cuda_device.h>
#include <primitiv/internal/cuda_utils.h>
#include <primitiv/device_ops/cuda/common.h>

namespace primitiv {
namespace devices {

void CUDA::random_log_normal_impl(float mean, float sd, Tensor &y) {
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateLogNormal(
        state_->curand.get(), MDATA(y), y.shape().size(), mean, sd));
}

}  // namespace devices
}  // namespace primitiv
