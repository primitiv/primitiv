#include <primitiv/config.h>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

EIGEN_DEV_FW_X(sigmoid, .5 + .5 * (.5 * x).tanh());
EIGEN_DEV_BW_X(sigmoid, gy * y * (1. - y));

}  // namespace devices
}  // namespace primitiv
