#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16DEV_KERNEL_FW_X(cos, ::cosf(px[i]));
CUDA16DEV_KERNEL_BW_X(cos, -::sinf(px[i]) * pgy[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16DEV_FW_X(cos);
CUDA16DEV_BW_X(cos);

}  // namespace devices
}  // namespace primitiv
