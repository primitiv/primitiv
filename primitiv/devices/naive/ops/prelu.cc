#include <primitiv/config.h>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X_CONST(prelu, src[i] * ((src[i] > 0) + k * (src[i] <= 0)));
CPUDEV_BW_X_CONST(prelu, pgy[i] * ((px[i] > 0) + k * (px[i] <= 0)));

}  // namespace devices
}  // namespace primitiv
