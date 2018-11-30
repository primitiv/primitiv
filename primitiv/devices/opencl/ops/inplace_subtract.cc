#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::inplace_subtract_impl(const Tensor &x, Tensor &y) {
  if (!state_->inplace_subtract_kernel.initialized()) {
    state_->initialize_kernel(state_->inplace_subtract_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/inplace_subtract.inc"
    }), "inplace_subtract_kernel");
  }
  const std::uint32_t size = y.shape().volume();
  const std::uint32_t mbx = x.shape().has_batch();
  const std::uint32_t mby = y.shape().has_batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, state_->inplace_subtract_kernel.group_size()[0]);
  const std::uint32_t bs = std::max(x.shape().batch(), y.shape().batch());
  state_->inplace_subtract_kernel.kernel().setArg(0, CDATA(x));
  state_->inplace_subtract_kernel.kernel().setArg(1, size);
  state_->inplace_subtract_kernel.kernel().setArg(2, mbx);
  state_->inplace_subtract_kernel.kernel().setArg(3, mby);
  state_->inplace_subtract_kernel.kernel().setArg(4, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->inplace_subtract_kernel.kernel(), cl::NullRange,
      cl::NDRange(g1 * state_->inplace_subtract_kernel.group_size()[0], bs),
      cl::NDRange(state_->inplace_subtract_kernel.group_size()[0], 1));
}

}  // namespace devices
}  // namespace primitiv
