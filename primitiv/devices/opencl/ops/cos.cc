#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::cos_fw_impl(const Tensor &x, Tensor &y) {
  if (!state_->cos_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->cos_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/cos.inc"
    }), "cos_fw_kernel");
  }
  OPENCLDEV_FW_X(cos)
}

void OpenCL::cos_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) {
  if (!state_->cos_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->cos_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/cos.inc"
    }), "cos_bw_kernel");
  }
  OPENCLDEV_BW_X(cos)
}

}  // namespace devices
}  // namespace primitiv
