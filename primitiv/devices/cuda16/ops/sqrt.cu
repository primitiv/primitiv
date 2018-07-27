#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDA16_KERNEL_FW_X(sqrt, ::__fsqrt_rn(X_VAL));
CUDA16_KERNEL_BW_X(sqrt, .5f * GY_VAL / Y_VAL);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X(sqrt);
CUDA16_DEV_BW_X(sqrt);

}  // namespace devices
}  // namespace primitiv
