#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X(tan, x.tan());
EIGEN_DEV_BW_X(tan, gy * (1. + y * y));

}  // namespace devices
}  // namespace primitiv
