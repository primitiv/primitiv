#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDA16_KERNEL_FW_X(negate, -X_VAL);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16_DEV_FW_X(negate);

}  // namespace devices
}  // namespace primitiv
