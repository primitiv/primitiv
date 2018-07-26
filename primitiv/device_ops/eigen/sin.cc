#include <primitiv/config.h>

#include <primitiv/core/eigen_device.h>
#include <primitiv/device_ops/eigen/common.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X(sin, x.sin());
EIGEN_DEV_BW_X(sin, gy * x.cos());

}  // namespace devices
}  // namespace primitiv
