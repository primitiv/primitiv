#include <primitiv/config.h>

#include <cmath>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(log, std::log(src[i]));
CPUDEV_BW_X(log, pgy[i] / px[i]);

}  // namespace devices
}  // namespace primitiv
