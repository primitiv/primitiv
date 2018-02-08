#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

CUDA16DEV_KERNEL_FW_X(sigmoid, .5f + .5f * ::tanhf(.5f * px[i]));
CUDA16DEV_KERNEL_BW_X(sigmoid, py[i] * (1.f - py[i]) * pgy[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDA16DEV_FW_X(sigmoid);
CUDA16DEV_BW_X(sigmoid);

}  // namespace devices
}  // namespace primitiv
