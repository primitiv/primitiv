#include <primitiv/config.h>

#include <primitiv/core/cuda16_device.h>
#include <primitiv/internal/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16_KERNEL_FW_X(tanh, ::tanhf(X_VAL));
CUDA16_KERNEL_BW_X(tanh, (1.f - Y_VAL * Y_VAL) * GY_VAL);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X(tanh);
CUDA16_DEV_BW_X(tanh);

}  // namespace devices
}  // namespace primitiv
