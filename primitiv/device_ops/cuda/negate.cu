#include <primitiv/config.h>

#include <primitiv/cuda_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda/common.h>

namespace {

CUDADEV_KERNEL_FW_X(negate, -px[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X(negate);

}  // namespace devices
}  // namespace primitiv
