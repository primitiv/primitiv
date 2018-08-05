#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X(cos, x.cos());
EIGEN_DEV_BW_X(cos, -gy * x.sin());

}  // namespace devices
}  // namespace primitiv
