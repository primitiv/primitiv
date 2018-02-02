#include <primitiv/config.h>

#include <cmath>

#include <primitiv/naive_device.h>
#include <primitiv/device_ops/naive_utils.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(exp, std::exp(src[i]));
CPUDEV_BW_X(exp, py[i] * pgy[i]);

}  // namespace devices
}  // namespace primitiv
