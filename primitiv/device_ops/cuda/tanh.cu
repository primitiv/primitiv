#include <primitiv/config.h>

#include <primitiv/cuda_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda/common.h>

namespace {

CUDADEV_KERNEL_FW_X(tanh, ::tanhf(px[i]));
CUDADEV_KERNEL_BW_X(tanh, (1.f - py[i] * py[i]) * pgy[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X(tanh);
CUDADEV_BW_X(tanh);

}  // namespace devices
}  // namespace primitiv
