#include <primitiv/config.h>

#include <primitiv/core/eigen_device.h>
#include <primitiv/device_ops/eigen/common.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X_CONST(prelu, (x > 0.).select(x, k * x));
EIGEN_DEV_BW_X_CONST(prelu, (x > 0.).select(gy, k * gy));

}  // namespace devices
}  // namespace primitiv
