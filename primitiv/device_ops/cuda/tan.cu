#include <primitiv/config.h>

#include <primitiv/cuda_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda/common.h>

namespace {

CUDADEV_KERNEL_FW_X(tan, ::tanf(px[i]));
CUDADEV_KERNEL_BW_X(tan, (1.f + py[i] * py[i]) * pgy[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X(tan);
CUDADEV_BW_X(tan);

}  // namespace devices
}  // namespace primitiv
