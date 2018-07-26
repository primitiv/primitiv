#include <primitiv/config.h>

#include <cmath>

#include <primitiv/core/naive_device.h>
#include <primitiv/device_ops/naive/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(tan, std::tan(src[i]));
CPUDEV_BW_X(tan, (1 + py[i] * py[i]) * pgy[i]);

}  // namespace devices
}  // namespace primitiv
