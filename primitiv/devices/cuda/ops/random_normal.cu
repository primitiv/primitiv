#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace primitiv {
namespace devices {

void CUDA::random_normal_impl(float mean, float sd, Tensor &y) {
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateNormal(
        state_->curand.get(), MDATA(y), y.shape().size(), mean, sd));
}

}  // namespace devices
}  // namespace primitiv
