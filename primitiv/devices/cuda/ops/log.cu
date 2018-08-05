#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDADEV_KERNEL_FW_X(log, ::logf(px[i]));
CUDADEV_KERNEL_BW_X(log, pgy[i] / px[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X(log);
CUDADEV_BW_X(log);

}  // namespace devices
}  // namespace primitiv
