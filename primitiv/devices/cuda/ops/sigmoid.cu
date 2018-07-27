#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

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
