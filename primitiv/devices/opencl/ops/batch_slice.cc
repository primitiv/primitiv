#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::batch_slice_fw_impl(
    const Tensor &x, std::uint32_t offset, Tensor &y) {
  if (!state_->batch_slice_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->batch_slice_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/batch_slice.inc"
    }), "batch_slice_fw_kernel");
  }
  const std::uint32_t volume = y.shape().volume();
  const std::uint32_t shift = volume * offset;
  const std::uint32_t size = y.shape().size();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->batch_slice_fw_kernel.group_size()[0]);
  state_->batch_slice_fw_kernel.kernel().setArg(0, CDATA(x));
  state_->batch_slice_fw_kernel.kernel().setArg(1, shift);
  state_->batch_slice_fw_kernel.kernel().setArg(2, size);
  state_->batch_slice_fw_kernel.kernel().setArg(3, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->batch_slice_fw_kernel.kernel(), cl::NullRange,
      cl::NDRange(num_blocks * state_->batch_slice_fw_kernel.group_size()[0]),
      cl::NDRange(state_->batch_slice_fw_kernel.group_size()[0]));
}

void OpenCL::batch_slice_bw_impl(const Tensor &gy, std::uint32_t offset, Tensor &gx) {
  if (!state_->batch_slice_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->batch_slice_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/batch_slice.inc"
    }), "batch_slice_bw_kernel");
  }
  const std::uint32_t volume = gy.shape().volume();
  const std::uint32_t shift = volume * offset;
  const std::uint32_t size = gy.shape().size();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, state_->batch_slice_bw_kernel.group_size()[0]);
  state_->batch_slice_bw_kernel.kernel().setArg(0, CDATA(gy));
  state_->batch_slice_bw_kernel.kernel().setArg(1, size);
  state_->batch_slice_bw_kernel.kernel().setArg(2, MDATA(gx));
  state_->batch_slice_bw_kernel.kernel().setArg(3, shift);
  state_->queue.enqueueNDRangeKernel(
      state_->batch_slice_bw_kernel.kernel(), cl::NullRange,
      cl::NDRange(g1 * state_->batch_slice_bw_kernel.group_size()[0]),
      cl::NDRange(state_->batch_slice_bw_kernel.group_size()[0]));
}

}  // namespace devices
}  // namespace primitiv
