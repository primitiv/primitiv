#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace primitiv {
namespace devices {

std::shared_ptr<void> CUDA16::new_handle(const Shape &shape) {
  // NOTE(odashi):
  // Always acquires multiple-of-2 values to enable packed operation.
  std::size_t size = shape.size();
  size += size & 1;
  return state_->pool.allocate(sizeof(half) * size);
}

}  // namespace devices
}  // namespace primitiv
