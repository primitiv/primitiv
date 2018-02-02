#include <primitiv/config.h>

#include <cmath>

#include <primitiv/naive_device.h>
#include <primitiv/device_ops/naive_utils.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(tanh, std::tanh(src[i]));
CPUDEV_BW_X(tanh, (1. - py[i] * py[i]) * pgy[i]);

}  // namespace devices
}  // namespace primitiv
