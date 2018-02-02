#include <primitiv/config.h>

#include <cmath>

#include <primitiv/naive_device.h>
#include <primitiv/device_ops/naive_utils.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(log, std::log(src[i]));
CPUDEV_BW_X(log, pgy[i] / px[i]);

}  // namespace devices
}  // namespace primitiv
