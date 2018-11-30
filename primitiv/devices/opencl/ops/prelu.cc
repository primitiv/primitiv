#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::prelu_fw_impl(const Tensor &x, float k, Tensor &y) {
  if (!state_->prelu_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->prelu_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/prelu.inc"
    }), "prelu_fw_kernel");
  }
  OPENCLDEV_FW_X_CONST(prelu);
}

void OpenCL::prelu_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) {
  if (!state_->prelu_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->prelu_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/prelu.inc"
    }), "prelu_bw_kernel");
  }
  OPENCLDEV_BW_X_CONST(prelu);
}

}  // namespace devices
}  // namespace primitiv
