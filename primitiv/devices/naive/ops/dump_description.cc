#include <primitiv/config.h>

#include <iostream>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

void Naive::dump_description() const {
  std::cerr << "Device " << this << std::endl;
  std::cerr << "  Type: Naive" << std::endl;
}

}  // namespace devices
}  // namespace primitiv
