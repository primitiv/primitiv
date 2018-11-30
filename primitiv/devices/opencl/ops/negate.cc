#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::negate_fw_impl(const Tensor &x, Tensor &y) {
  if (!state_->negate_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->negate_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/negate.inc"
    }), "negate_fw_kernel");
  }
  OPENCLDEV_FW_X(negate)
}

}  // namespace devices
}  // namespace primitiv
