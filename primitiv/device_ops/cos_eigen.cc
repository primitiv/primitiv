#include <primitiv/config.h>

#include <primitiv/eigen_device.h>
#include <primitiv/device_ops/eigen_utils.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X(cos, x.cos());
EIGEN_DEV_BW_X(cos, -gy * x.sin());

}  // namespace devices
}  // namespace primitiv
