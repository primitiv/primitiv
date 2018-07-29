#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDADEV_KERNEL_FW_X(negate, -px[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X(negate);

}  // namespace devices
}  // namespace primitiv
