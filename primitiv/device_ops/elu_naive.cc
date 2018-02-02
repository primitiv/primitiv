#include <primitiv/config.h>

#include <primitiv/naive_device.h>
#include <primitiv/device_ops/naive_utils.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X_CONST(
    elu, src[i] * (src[i] > 0) + k * (std::exp(src[i] * (src[i] <= 0)) - 1));
CPUDEV_BW_X_CONST(elu, pgy[i] * ((px[i] > 0) + (py[i] + k) * (px[i] <= 0)));

}  // namespace devices
}  // namespace primitiv
