#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::tan_fw_impl(const Tensor &x, Tensor &y) {
  if (!state_->tan_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->tan_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/tan.inc"
    }), "tan_fw_kernel");
  }
  OPENCLDEV_FW_X(tan)
}

void OpenCL::tan_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) {
  if (!state_->tan_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->tan_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/tan.inc"
    }), "tan_bw_kernel");
  }
  OPENCLDEV_BW_X(tan)
}

}  // namespace devices
}  // namespace primitiv
