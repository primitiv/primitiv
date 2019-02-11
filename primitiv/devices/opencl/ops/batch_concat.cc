#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::batch_concat_fw_impl(
    const std::vector<const Tensor *> &xs, Tensor &y) {
  std::uint32_t offset = 0;
  for (const Tensor *x : xs) {
    const std::uint32_t span = x->shape().size();
    const std::uint32_t num_blocks = ::calc_num_blocks(
        span, state_->batch_concat_fw_group_size);
    state_->batch_concat_fw_kernel.setArg(0, CDATA(*x));
    state_->batch_concat_fw_kernel.setArg(1, span);
    state_->batch_concat_fw_kernel.setArg(2, MDATA(y));
    state_->batch_concat_fw_kernel.setArg(3, offset);
    state_->queue.enqueueNDRangeKernel(
        state_->batch_concat_fw_kernel, cl::NullRange,
        cl::NDRange(num_blocks * state_->batch_concat_fw_group_size),
        cl::NDRange(state_->batch_concat_fw_group_size), nullptr, nullptr);
    offset += span;
  }
}

}  // namespace devices
}  // namespace primitiv
