#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::elu_fw_impl(const Tensor &x, float k, Tensor &y) {
  if (!state_->elu_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->elu_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/elu.inc"
    }), "elu_fw_kernel");
  }
  OPENCLDEV_FW_X_CONST(elu);
}

void OpenCL::elu_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) {
  if (!state_->elu_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->elu_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/elu.inc"
    }), "elu_bw_kernel");
  }
  OPENCLDEV_BW_X_CONST(elu);
}

}  // namespace devices
}  // namespace primitiv
