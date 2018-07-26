#include <primitiv/config.h>

#include <cmath>

#include <primitiv/core/naive_device.h>
#include <primitiv/device_ops/naive/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(sqrt, std::sqrt(src[i]));
CPUDEV_BW_X(sqrt, .5 * pgy[i] / py[i]);

}  // namespace devices
}  // namespace primitiv
