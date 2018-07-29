#include <primitiv/config.h>

#include <cmath>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(sin, std::sin(src[i]));
CPUDEV_BW_X(sin, std::cos(px[i]) * pgy[i]);

}  // namespace devices
}  // namespace primitiv
