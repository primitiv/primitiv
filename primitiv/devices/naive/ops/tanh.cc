#include <primitiv/config.h>

#include <cmath>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(tanh, std::tanh(src[i]));
CPUDEV_BW_X(tanh, (1. - py[i] * py[i]) * pgy[i]);

}  // namespace devices
}  // namespace primitiv
