#include <primitiv/config.h>

#include <primitiv/cuda_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/common_cuda.h>

namespace primitiv {
namespace devices {

std::shared_ptr<void> CUDA::new_handle(const Shape &shape) {
  return state_->pool.allocate(sizeof(float) * shape.size());
}

}  // namespace devices
}  // namespace primitiv
