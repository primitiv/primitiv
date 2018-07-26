#include <primitiv/config.h>

#include <cmath>

#include <primitiv/core/naive_device.h>
#include <primitiv/device_ops/naive/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(log, std::log(src[i]));
CPUDEV_BW_X(log, pgy[i] / px[i]);

}  // namespace devices
}  // namespace primitiv
