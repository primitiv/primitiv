#include <primitiv/config.h>

#include <cmath>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(sqrt, std::sqrt(src[i]));
CPUDEV_BW_X(sqrt, .5 * pgy[i] / py[i]);

}  // namespace devices
}  // namespace primitiv
