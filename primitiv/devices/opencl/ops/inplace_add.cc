#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::inplace_add_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t size = y.shape().volume();
  const std::uint32_t mbx = x.shape().has_batch();
  const std::uint32_t mby = y.shape().has_batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, state_->inplace_add_group_size);
  const std::uint32_t bs = std::max(x.shape().batch(), y.shape().batch());
  state_->inplace_add_kernel.setArg(0, CDATA(x));
  state_->inplace_add_kernel.setArg(1, size);
  state_->inplace_add_kernel.setArg(2, mbx);
  state_->inplace_add_kernel.setArg(3, mby);
  state_->inplace_add_kernel.setArg(4, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->inplace_add_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->inplace_add_group_size, bs, 1),
      cl::NDRange(state_->inplace_add_group_size, 1, 1));
}

}  // namespace devices
}  // namespace primitiv
