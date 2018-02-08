#include <primitiv/config.h>

#include <iostream>

#include <primitiv/eigen_device.h>
#include <primitiv/device_ops/eigen/common.h>

namespace primitiv {
namespace devices {

void Eigen::dump_description() const {
  std::cerr << "Device " << this << std::endl;
  std::cerr << "  Type: Eigen" << std::endl;
}

}  // namespace devices
}  // namespace primitiv
