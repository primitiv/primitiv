#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::inplace_multiply_const_impl(float k, Tensor &x) {
  const std::uint32_t size = x.shape().size();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, state_->inplace_multiply_const_group_size);
  state_->inplace_multiply_const_kernel.setArg(0, k);
  state_->inplace_multiply_const_kernel.setArg(1, size);
  state_->inplace_multiply_const_kernel.setArg(2, MDATA(x));
  state_->queue.enqueueNDRangeKernel(
      state_->inplace_multiply_const_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->inplace_multiply_const_group_size),
      cl::NDRange(state_->inplace_multiply_const_group_size));
}

}  // namespace devices
}  // namespace primitiv
