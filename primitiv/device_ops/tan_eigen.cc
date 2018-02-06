#include <primitiv/config.h>

#include <primitiv/eigen_device.h>
#include <primitiv/device_ops/common_eigen.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X(tan, x.tan());
EIGEN_DEV_BW_X(tan, gy * (1. + y * y));

}  // namespace devices
}  // namespace primitiv
