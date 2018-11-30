#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::log_fw_impl(const Tensor &x, Tensor &y) {
  if (!state_->log_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->log_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/log.inc"
    }), "log_fw_kernel");
  }
  OPENCLDEV_FW_X(log)
}

void OpenCL::log_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) {
  if (!state_->log_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->log_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/log.inc"
    }), "log_bw_kernel");
  }
  OPENCLDEV_BW_X(log)
}

}  // namespace devices
}  // namespace primitiv
