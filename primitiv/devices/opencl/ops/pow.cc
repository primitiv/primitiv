#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

OPENCLDEV_FW_X_CONST(pow_const_r);
OPENCLDEV_FW_X_CONST(pow_const_l);
OPENCLDEV_BW_X_CONST(pow_const_r);
OPENCLDEV_BW_X_CONST(pow_const_l);
OPENCLDEV_FW_X_SCALAR(pow_scalar_r);
OPENCLDEV_FW_X_SCALAR(pow_scalar_l);
OPENCLDEV_FW_AB(pow);
OPENCLDEV_BW_AB(pow);

}  // namespace devices
}  // namespace primitiv
