#include <primitiv/config.h>

#include <cmath>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(abs, std::abs(src[i]));
CPUDEV_BW_X(abs, ((px[i] > 0) - (px[i] < 0)) * pgy[i]);

}  // namespace devices
}  // namespace primitiv
