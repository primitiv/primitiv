#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

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
