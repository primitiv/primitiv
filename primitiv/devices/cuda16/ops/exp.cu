#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDA16_KERNEL_FW_X(exp, ::expf(X_VAL));
CUDA16_KERNEL_BW_X(exp, Y_VAL * GY_VAL);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X(exp);
CUDA16_DEV_BW_X(exp);

}  // namespace devices
}  // namespace primitiv
