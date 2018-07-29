#include <primitiv/config.h>

#include <cmath>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(cos, std::cos(src[i]));
CPUDEV_BW_X(cos, -std::sin(px[i]) * pgy[i]);

}  // namespace devices
}  // namespace primitiv
