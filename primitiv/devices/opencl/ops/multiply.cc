#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

OPENCLDEV_FW_X_CONST(multiply_const);
OPENCLDEV_BW_X_CONST(multiply_const);
OPENCLDEV_FW_X_SCALAR(multiply_scalar);
OPENCLDEV_FW_AB(multiply);
OPENCLDEV_BW_AB(multiply);

}  // namespace devices
}  // namespace primitiv
