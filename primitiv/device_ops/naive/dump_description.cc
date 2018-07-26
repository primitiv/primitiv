#include <primitiv/config.h>

#include <iostream>

#include <primitiv/core/naive_device.h>
#include <primitiv/device_ops/naive/common.h>

namespace primitiv {
namespace devices {

void Naive::dump_description() const {
  std::cerr << "Device " << this << std::endl;
  std::cerr << "  Type: Naive" << std::endl;
}

}  // namespace devices
}  // namespace primitiv
