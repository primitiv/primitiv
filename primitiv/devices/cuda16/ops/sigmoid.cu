#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDA16_KERNEL_FW_X(sigmoid, .5f + .5f * ::tanhf(.5f * X_VAL));
CUDA16_KERNEL_BW_X(sigmoid, Y_VAL * (1.f - Y_VAL) * GY_VAL);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X(sigmoid);
CUDA16_DEV_BW_X(sigmoid);

}  // namespace devices
}  // namespace primitiv
