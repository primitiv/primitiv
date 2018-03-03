#include <primitiv/config.h>

#include <cstdlib>

#include <primitiv/eigen_device.h>
#include <primitiv/device_ops/eigen/common.h>

namespace primitiv {
namespace devices {

std::shared_ptr<void> Eigen::new_handle(const Shape &shape) {
  const std::uint32_t mem_size = sizeof(float) * shape.size();
  void *data = std::malloc(mem_size);
  if (!data) {
    PRIMITIV_THROW_ERROR("Memory allocation failed. Requested size: " << mem_size);
  }
  return std::shared_ptr<void>(data, std::free);
}

}  // namespace devices
}  // namespace primitiv
