#include <primitiv/config.h>

#include <cmath>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(tan, std::tan(src[i]));
CPUDEV_BW_X(tan, (1 + py[i] * py[i]) * pgy[i]);

}  // namespace devices
}  // namespace primitiv
