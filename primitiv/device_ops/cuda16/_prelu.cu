#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16DEV_KERNEL_FW_X_CONST(prelu, ::fmaxf(px[i], .0f) + k * ::fminf(px[i], .0f));
CUDA16DEV_KERNEL_BW_X_CONST(prelu, pgy[i] * ((px[i] > .0f) + k * (px[i] <= .0f)));

}  // namespace

namespace primitiv {
namespace devices {

CUDA16DEV_FW_X_CONST(prelu);
CUDA16DEV_BW_X_CONST(prelu);

}  // namespace devices
}  // namespace primitiv
