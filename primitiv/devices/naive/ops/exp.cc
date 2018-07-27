#include <primitiv/config.h>

#include <cmath>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(exp, std::exp(src[i]));
CPUDEV_BW_X(exp, py[i] * pgy[i]);

}  // namespace devices
}  // namespace primitiv
