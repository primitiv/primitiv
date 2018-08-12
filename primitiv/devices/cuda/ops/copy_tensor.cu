#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace primitiv {
namespace devices {

void CUDA::copy_tensor_impl(const Tensor &x, Tensor &y) {
  switch (x.device().type()) {
    case DeviceType::NAIVE:
      reset_tensor_by_array(CDATA(x), y);
      break;
    case DeviceType::CUDA:
      CUDA_CALL(::cudaSetDevice(dev_id_));
      // NOTE(odashi):
      // If source/destination devices use the unified memory space on the 64
      // bits machine, we can perform ::cudaMemcpy to copy data beyond devices.
      CUDA_CALL(::cudaMemcpyAsync(
            MDATA(y), CDATA(x),
            sizeof(float) * x.shape().size(),
            cudaMemcpyDeviceToDevice, 0));
      break;
    //case DeviceType::CUDA16:
      // TODO(odashi): Implement this section.
    default:
      reset_tensor_by_vector(x.to_vector(), y);
  }
}

}  // namespace devices
}  // namespace primitiv
