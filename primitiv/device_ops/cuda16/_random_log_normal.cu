#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace primitiv {
namespace devices {

void CUDA16::random_log_normal_impl(float mean, float sd, Tensor &y) {
  THROW_NOT_IMPLEMENTED;
  /*CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateLogNormal(
        state_->curand.get(), MDATA(float, y), y.shape().size(), mean, sd));
*/}

}  // namespace devices
}  // namespace primitiv
