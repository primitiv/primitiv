#include <primitiv/config.h>

#include <primitiv/cuda_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda/common.h>

namespace {

CUDADEV_KERNEL_FW_X(sigmoid, .5f + .5f * ::tanhf(.5f * px[i]));
CUDADEV_KERNEL_BW_X(sigmoid, py[i] * (1.f - py[i]) * pgy[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X(sigmoid);
CUDADEV_BW_X(sigmoid);

}  // namespace devices
}  // namespace primitiv
