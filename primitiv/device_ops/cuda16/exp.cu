#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

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
