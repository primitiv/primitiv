#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X_CONST(elu, (x > 0.).select(x, k * (x.exp() - 1.)));
EIGEN_DEV_BW_X_CONST(elu, (x > 0.).select(gy, (y + k) * gy));

}  // namespace devices
}  // namespace primitiv
