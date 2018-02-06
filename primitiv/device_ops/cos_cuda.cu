#include <primitiv/config.h>

#include <primitiv/cuda_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/common_cuda.h>

namespace {

CUDADEV_KERNEL_FW_X(cos, ::cosf(px[i]));
CUDADEV_KERNEL_BW_X(cos, -::sinf(px[i]) * pgy[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X(cos);
CUDADEV_BW_X(cos);

}  // namespace devices
}  // namespace primitiv
