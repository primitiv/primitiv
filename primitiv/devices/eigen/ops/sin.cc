#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X(sin, x.sin());
EIGEN_DEV_BW_X(sin, gy * x.cos());

}  // namespace devices
}  // namespace primitiv
