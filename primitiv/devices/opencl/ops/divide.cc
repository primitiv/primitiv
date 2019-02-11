#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

OPENCLDEV_FW_X_CONST(divide_const_r);
OPENCLDEV_FW_X_CONST(divide_const_l);
OPENCLDEV_BW_X_CONST(divide_const_r);
OPENCLDEV_BW_X_CONST(divide_const_l);
OPENCLDEV_FW_X_SCALAR(divide_scalar_r);
OPENCLDEV_FW_X_SCALAR(divide_scalar_l);
OPENCLDEV_FW_AB(divide);
OPENCLDEV_BW_AB(divide);

}  // namespace devices
}  // namespace primitiv
