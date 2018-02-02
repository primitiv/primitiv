#include <primitiv/config.h>

#include <primitiv/naive_device.h>
#include <primitiv/device_ops/naive_utils.h>

namespace primitiv {
namespace devices {

CPUDEV_FW_X(negate, -src[i]);

}  // namespace devices
}  // namespace primitiv
