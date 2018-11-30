#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::sigmoid_fw_impl(const Tensor &x, Tensor &y) {
  if (!state_->sigmoid_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->sigmoid_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/sigmoid.inc"
    }), "sigmoid_fw_kernel");
  }
  OPENCLDEV_FW_X(sigmoid)
}

void OpenCL::sigmoid_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) {
  if (!state_->sigmoid_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->sigmoid_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/sigmoid.inc"
    }), "sigmoid_bw_kernel");
  }
  OPENCLDEV_BW_X(sigmoid)
}

}  // namespace devices
}  // namespace primitiv
