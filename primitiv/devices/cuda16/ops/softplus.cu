#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDA16_KERNEL_FW_X(
    softplus, ::fmaxf(X_VAL, .0f) + ::logf(1.f + ::expf(-::fabs(X_VAL))));
CUDA16_KERNEL_BW_X(softplus, (.5f + .5f * ::tanhf(.5f * X_VAL)) * GY_VAL);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X(softplus);
CUDA16_DEV_BW_X(softplus);

}  // namespace devices
}  // namespace primitiv
