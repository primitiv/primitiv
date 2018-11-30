#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::batch_sum_fw_impl(const Tensor &x, Tensor &y) {
  if (!state_->batch_sum_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->batch_sum_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/batch_sum.inc"
    }), "batch_sum_fw_kernel");
  }
  const std::uint32_t size = y.shape().size();
  const std::uint32_t batch = x.shape().batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, state_->batch_sum_fw_kernel.group_size()[0]);
  state_->batch_sum_fw_kernel.kernel().setArg(0, CDATA(x));
  state_->batch_sum_fw_kernel.kernel().setArg(1, size);
  state_->batch_sum_fw_kernel.kernel().setArg(2, batch);
  state_->batch_sum_fw_kernel.kernel().setArg(3, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->batch_sum_fw_kernel.kernel(), cl::NullRange,
      cl::NDRange(g1 * state_->batch_sum_fw_kernel.group_size()[0]),
      cl::NDRange(state_->batch_sum_fw_kernel.group_size()[0]));
}

}  // namespace devices
}  // namespace primitiv
