#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>

namespace primitiv {
namespace devices {

std::shared_ptr<void> OpenCL::new_handle(const Shape &shape) {
  return state_->pool.allocate(sizeof(float) * shape.size());
}

}  // namespace devices
}  // namespace primitiv
