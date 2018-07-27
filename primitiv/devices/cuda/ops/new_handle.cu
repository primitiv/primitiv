#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace primitiv {
namespace devices {

std::shared_ptr<void> CUDA::new_handle(const Shape &shape) {
  return state_->pool.allocate(sizeof(float) * shape.size());
}

}  // namespace devices
}  // namespace primitiv
