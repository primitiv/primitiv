#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDA16_KERNEL_FW_X_CONST(
    elu, ::fmaxf(X_VAL, .0f) + k * (::expf(::fminf(X_VAL, .0f)) - 1.0f));
CUDA16_KERNEL_BW_X_CONST(
    elu, GY_VAL * ((X_VAL > .0f) + (Y_VAL + k) * (X_VAL <= .0f)));

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X_CONST(elu);
CUDA16_DEV_BW_X_CONST(elu);

}  // namespace devices
}  // namespace primitiv
