#include <primitiv/config.h>

#include <cmath>

#include <primitiv/core/naive_device.h>
#include <primitiv/device_ops/naive/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(sigmoid, .5 + .5 * std::tanh(.5 * src[i]));
CPUDEV_BW_X(sigmoid, py[i] * (1. - py[i]) * pgy[i]);

}  // namespace devices
}  // namespace primitiv
