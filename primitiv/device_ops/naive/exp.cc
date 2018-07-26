#include <primitiv/config.h>

#include <cmath>

#include <primitiv/core/naive_device.h>
#include <primitiv/device_ops/naive/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(exp, std::exp(src[i]));
CPUDEV_BW_X(exp, py[i] * pgy[i]);

}  // namespace devices
}  // namespace primitiv
