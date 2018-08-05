#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDADEV_KERNEL_FW_X(sin, ::sinf(px[i]));
CUDADEV_KERNEL_BW_X(sin, ::cosf(px[i]) * pgy[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X(sin);
CUDADEV_BW_X(sin);

}  // namespace devices
}  // namespace primitiv
