#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDADEV_KERNEL_FW_X_CONST(
    elu, ::fmaxf(px[i], .0f) + k * (::expf(::fminf(px[i], .0f)) - 1.0f));
CUDADEV_KERNEL_BW_X_CONST(
    elu, pgy[i] * ((px[i] > .0f) + (py[i] + k) * (px[i] <= .0f)));

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X_CONST(elu);
CUDADEV_BW_X_CONST(elu);

}  // namespace devices
}  // namespace primitiv
