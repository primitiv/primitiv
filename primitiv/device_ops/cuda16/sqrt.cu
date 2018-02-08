#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16DEV_KERNEL_FW_X(sqrt, ::__fsqrt_rn(px[i]));
CUDA16DEV_KERNEL_BW_X(sqrt, .5f * pgy[i] / py[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16DEV_FW_X(sqrt);
CUDA16DEV_BW_X(sqrt);

}  // namespace devices
}  // namespace primitiv
