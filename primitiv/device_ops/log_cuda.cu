#include <primitiv/config.h>

#include <primitiv/cuda_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/common_cuda.h>

namespace {

CUDADEV_KERNEL_FW_X(log, ::logf(px[i]));
CUDADEV_KERNEL_BW_X(log, pgy[i] / px[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X(log);
CUDADEV_BW_X(log);

}  // namespace devices
}  // namespace primitiv
