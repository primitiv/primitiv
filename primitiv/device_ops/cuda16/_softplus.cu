#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16DEV_KERNEL_FW_X(
    softplus, ::fmaxf(px[i], .0f) + ::logf(1.f + ::expf(-::fabs(px[i]))));
CUDA16DEV_KERNEL_BW_X(softplus, (.5f + .5f * ::tanhf(.5f * px[i])) * pgy[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16DEV_FW_X(softplus);
CUDA16DEV_BW_X(softplus);

}  // namespace devices
}  // namespace primitiv
