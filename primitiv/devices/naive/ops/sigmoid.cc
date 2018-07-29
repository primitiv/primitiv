#include <primitiv/config.h>

#include <cmath>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(sigmoid, .5 + .5 * std::tanh(.5 * src[i]));
CPUDEV_BW_X(sigmoid, py[i] * (1. - py[i]) * pgy[i]);

}  // namespace devices
}  // namespace primitiv
