#include <primitiv/config.h>

#include <primitiv/core/eigen_device.h>
#include <primitiv/device_ops/eigen/common.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X(exp, x.exp());
EIGEN_DEV_BW_X(exp, gy * y);

}  // namespace devices
}  // namespace primitiv
