#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/internal/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16_KERNEL_FW_X(log, ::logf(X_VAL));
CUDA16_KERNEL_BW_X(log, GY_VAL / X_VAL);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X(log);
CUDA16_DEV_BW_X(log);

}  // namespace devices
}  // namespace primitiv
