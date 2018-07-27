#include <primitiv/config.h>

#include <cstdlib>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

std::shared_ptr<void> Naive::new_handle(const Shape &shape) {
  const std::uint32_t mem_size = sizeof(float) * shape.size();
  void *data = std::malloc(mem_size);
  if (!data) {
    PRIMITIV_THROW_ERROR("Memory allocation failed. Requested size: " << mem_size);
  }
  return std::shared_ptr<void>(data, std::free);
}

}  // namespace devices
}  // namespace primitiv
