#include <primitiv/config.h>

#include <iostream>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

void Eigen::dump_description() const {
  std::cerr << "Device " << this << std::endl;
  std::cerr << "  Type: Eigen" << std::endl;
}

}  // namespace devices
}  // namespace primitiv
