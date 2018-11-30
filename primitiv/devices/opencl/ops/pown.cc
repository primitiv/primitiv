#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::pown_fw_impl(const Tensor &x, std::int32_t k, Tensor &y) {
  if (!state_->pown_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->pown_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/pown.inc"
    }), "pown_fw_kernel");
  }
  const std::uint32_t size = x.shape().size();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->pown_fw_kernel.group_size()[0]);
  state_->pown_fw_kernel.kernel().setArg(0, CDATA(x));
  state_->pown_fw_kernel.kernel().setArg(1, k);
  state_->pown_fw_kernel.kernel().setArg(2, size);
  state_->pown_fw_kernel.kernel().setArg(3, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->pown_fw_kernel.kernel(), cl::NullRange,
      cl::NDRange(num_blocks * state_->pown_fw_kernel.group_size()[0]),
      cl::NDRange(state_->pown_fw_kernel.group_size()[0]));
}

void OpenCL::pown_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, std::int32_t k,
    Tensor &gx) {
  if (!state_->pown_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->pown_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/pown.inc"
    }), "pown_bw_kernel");
  }
  const std::uint32_t size = x.shape().size();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->pown_bw_kernel.group_size()[0]);
  state_->pown_bw_kernel.kernel().setArg(0, CDATA(x));
  state_->pown_bw_kernel.kernel().setArg(1, CDATA(y));
  state_->pown_bw_kernel.kernel().setArg(2, CDATA(gy));
  state_->pown_bw_kernel.kernel().setArg(3, k);
  state_->pown_bw_kernel.kernel().setArg(4, size);
  state_->pown_bw_kernel.kernel().setArg(5, MDATA(gx));
  state_->queue.enqueueNDRangeKernel(
      state_->pown_bw_kernel.kernel(), cl::NullRange,
      cl::NDRange(num_blocks * state_->pown_bw_kernel.group_size()[0]),
      cl::NDRange(state_->pown_bw_kernel.group_size()[0]));
}

}  // namespace devices
}  // namespace primitiv
