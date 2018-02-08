#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16DEV_KERNEL_FW_X(negate, -px[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16DEV_FW_X(negate);

}  // namespace devices
}  // namespace primitiv
