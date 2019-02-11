#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::pown_fw_impl(const Tensor &x, std::int32_t k, Tensor &y) {
  const std::uint32_t size = x.shape().size();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->pown_fw_group_size);
  state_->pown_fw_kernel.setArg(0, CDATA(x));
  state_->pown_fw_kernel.setArg(1, k);
  state_->pown_fw_kernel.setArg(2, size);
  state_->pown_fw_kernel.setArg(3, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->pown_fw_kernel, cl::NullRange,
      cl::NDRange(num_blocks * state_->pown_fw_group_size),
      cl::NDRange(state_->pown_fw_group_size));
}

void OpenCL::pown_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, std::int32_t k,
    Tensor &gx) {
  const std::uint32_t size = x.shape().size();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->pown_bw_group_size);
  state_->pown_bw_kernel.setArg(0, CDATA(x));
  state_->pown_bw_kernel.setArg(1, CDATA(y));
  state_->pown_bw_kernel.setArg(2, CDATA(gy));
  state_->pown_bw_kernel.setArg(3, k);
  state_->pown_bw_kernel.setArg(4, size);
  state_->pown_bw_kernel.setArg(5, MDATA(gx));
  state_->queue.enqueueNDRangeKernel(
      state_->pown_bw_kernel, cl::NullRange,
      cl::NDRange(num_blocks * state_->pown_bw_group_size),
      cl::NDRange(state_->pown_bw_group_size));
}

}  // namespace devices
}  // namespace primitiv
