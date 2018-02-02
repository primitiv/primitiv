#include <primitiv/config.h>

#include <primitiv/eigen_device.h>
#include <primitiv/device_ops/eigen_utils.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X(tanh, x.tanh());
EIGEN_DEV_BW_X(tanh, gy * (1. - y * y));

}  // namespace devices
}  // namespace primitiv
