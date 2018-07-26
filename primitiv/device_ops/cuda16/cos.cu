#include <primitiv/config.h>

#include <primitiv/core/cuda16_device.h>
#include <primitiv/internal/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16_KERNEL_FW_X(cos, ::cosf(X_VAL));
CUDA16_KERNEL_BW_X(cos, -::sinf(X_VAL) * GY_VAL);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X(cos);
CUDA16_DEV_BW_X(cos);

}  // namespace devices
}  // namespace primitiv
