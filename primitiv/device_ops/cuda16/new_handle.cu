#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace primitiv {
namespace devices {

std::shared_ptr<void> CUDA16::new_handle(const Shape &shape) {
  return state_->pool.allocate(sizeof(float) * shape.size());
}

}  // namespace devices
}  // namespace primitiv
