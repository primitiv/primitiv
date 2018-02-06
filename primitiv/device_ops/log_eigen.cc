#include <primitiv/config.h>

#include <primitiv/eigen_device.h>
#include <primitiv/device_ops/common_eigen.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X(log, x.log());
EIGEN_DEV_BW_X(log, gy / x);

}  // namespace devices
}  // namespace primitiv
