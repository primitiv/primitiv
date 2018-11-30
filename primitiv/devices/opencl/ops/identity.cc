#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::identity_impl(Tensor &y) {
  if (!state_->set_identity_kernel.initialized()) {
    state_->initialize_kernel(state_->set_identity_kernel, std::string({
#include "primitiv/devices/opencl/kernels/identity.inc"
    }), "set_identity_kernel");
  }
  const std::uint32_t size = y.shape().size();
  const std::uint32_t skip = y.shape()[0] + 1;
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->set_identity_kernel.group_size()[0]);
  state_->set_identity_kernel.kernel().setArg(0, size);
  state_->set_identity_kernel.kernel().setArg(1, skip);
  state_->set_identity_kernel.kernel().setArg(2, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->set_identity_kernel.kernel(), cl::NullRange,
      cl::NDRange(num_blocks * state_->set_identity_kernel.group_size()[0]),
      cl::NDRange(state_->set_identity_kernel.group_size()[0]));
}

}  // namespace devices
}  // namespace primitiv
