#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16DEV_KERNEL_FW_X(log, ::logf(px[i]));
CUDA16DEV_KERNEL_BW_X(log, pgy[i] / px[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16DEV_FW_X(log);
CUDA16DEV_BW_X(log);

}  // namespace devices
}  // namespace primitiv
