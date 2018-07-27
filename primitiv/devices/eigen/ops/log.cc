#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X(log, x.log());
EIGEN_DEV_BW_X(log, gy / x);

}  // namespace devices
}  // namespace primitiv
