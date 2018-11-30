#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::abs_fw_impl(const Tensor &x, Tensor &y) {
  if (!state_->abs_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->abs_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/abs.inc"
    }), "abs_fw_kernel");
  }
  OPENCLDEV_FW_X(abs)
}

void OpenCL::abs_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) {
  if (!state_->abs_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->abs_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/abs.inc"
    }), "abs_bw_kernel");
  }
  OPENCLDEV_BW_X(abs)
}

}  // namespace devices
}  // namespace primitiv
