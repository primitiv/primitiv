#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::sin_fw_impl(const Tensor &x, Tensor &y) {
  if (!state_->sin_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->sin_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/sin.inc"
    }), "sin_fw_kernel");
  }
  OPENCLDEV_FW_X(sin)
}

void OpenCL::sin_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) {
  if (!state_->sin_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->sin_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/sin.inc"
    }), "sin_bw_kernel");
  }
  OPENCLDEV_BW_X(sin)
}

}  // namespace devices
}  // namespace primitiv
