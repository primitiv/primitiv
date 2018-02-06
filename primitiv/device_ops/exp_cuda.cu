#include <primitiv/config.h>

#include <primitiv/cuda_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/common_cuda.h>

namespace {

CUDADEV_KERNEL_FW_X(exp, ::expf(px[i]));
CUDADEV_KERNEL_BW_X(exp, py[i] * pgy[i]);

}  // namespace

namespace primitiv {
namespace devices {

CUDADEV_FW_X(exp);
CUDADEV_BW_X(exp);

}  // namespace devices
}  // namespace primitiv
