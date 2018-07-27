#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDADEV_KERNEL_FW_X(cos, ::cosf(px[i]));
CUDADEV_KERNEL_BW_X(cos, -::sinf(px[i]) * pgy[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X(cos);
CUDADEV_BW_X(cos);

}  // namespace devices
}  // namespace primitiv
