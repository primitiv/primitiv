#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16_KERNEL_FW_X(negate, -X_VAL);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X(negate);

}  // namespace devices
}  // namespace primitiv
