#include <primitiv/config.h>

#include <iostream>

#include <primitiv/naive_device.h>
#include <primitiv/device_ops/common_naive.h>

namespace primitiv {
namespace devices {

void Naive::dump_description() const {
  std::cerr << "Device " << this << std::endl;
  std::cerr << "  Type: Naive" << std::endl;
}

}  // namespace devices
}  // namespace primitiv
