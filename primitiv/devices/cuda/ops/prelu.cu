#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

CUDADEV_KERNEL_FW_X_CONST(prelu, ::fmaxf(px[i], .0f) + k * ::fminf(px[i], .0f));
CUDADEV_KERNEL_BW_X_CONST(prelu, pgy[i] * ((px[i] > .0f) + k * (px[i] <= .0f)));

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X_CONST(prelu);
CUDADEV_BW_X_CONST(prelu);

}  // namespace devices
}  // namespace primitiv
