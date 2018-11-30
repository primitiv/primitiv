#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::tanh_fw_impl(const Tensor &x, Tensor &y) {
  if (!state_->tanh_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->tanh_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/tanh.inc"
    }), "tanh_fw_kernel");
  }
  OPENCLDEV_FW_X(tanh);
}

void OpenCL::tanh_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) {
  if (!state_->tanh_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->tanh_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/tanh.inc"
    }), "tanh_bw_kernel");
  }
  OPENCLDEV_BW_X(tanh);
}

}  // namespace devices
}  // namespace primitiv
