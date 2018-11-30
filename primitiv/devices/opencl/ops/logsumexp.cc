#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::logsumexp_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  if (!state_->logsumexp_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->logsumexp_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/logsumexp.inc"
    }), "logsumexp_fw_kernel");
  }
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  state_->logsumexp_fw_kernel.kernel().setArg(0, CDATA(x));
  state_->logsumexp_fw_kernel.kernel().setArg(1, s);
  state_->logsumexp_fw_kernel.kernel().setArg(2, n);
  state_->logsumexp_fw_kernel.kernel().setArg(3, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->logsumexp_fw_kernel.kernel(), cl::NullRange,
      cl::NDRange(r * state_->logsumexp_fw_kernel.group_size()[0]),
      cl::NDRange(state_->logsumexp_fw_kernel.group_size()[0]));
}

}  // namespace devices
}  // namespace primitiv
