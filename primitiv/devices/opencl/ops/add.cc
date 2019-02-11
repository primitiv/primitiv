#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

OPENCLDEV_FW_X_CONST(add_const);
OPENCLDEV_BW_X_CONST(add_const);
OPENCLDEV_FW_X_SCALAR(add_scalar);
OPENCLDEV_FW_AB(add);
OPENCLDEV_BW_AB(add);

}  // namespace devices
}  // namespace primitiv
