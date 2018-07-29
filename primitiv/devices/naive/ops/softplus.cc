#include <primitiv/config.h>

#include <cmath>

#include <primitiv/devices/naive/device.h>
#include <primitiv/devices/naive/ops/common.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(
    softplus, src[i] > 0
      ? src[i] + std::log(1 + std::exp(-src[i]))
      : std::log(1 + std::exp(src[i])));
CPUDEV_BW_X(softplus, (.5 + .5 * std::tanh(.5 * px[i])) * pgy[i]);

}  // namespace devices
}  // namespace primitiv
