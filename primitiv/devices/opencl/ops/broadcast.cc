#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::broadcast_fw_impl(
    const Tensor &x, std::uint32_t dim, std::uint32_t size, Tensor &y) {
  if (!state_->broadcast_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->broadcast_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/broadcast.inc"
    }), "broadcast_fw_kernel");
  }
  const std::uint32_t skip1 = y.shape().lower_volume(dim);
  const std::uint32_t skip2 = skip1 * size;
  const std::uint32_t total = y.shape().size();
  const std::uint32_t g1 = ::calc_num_blocks(
      total, state_->broadcast_fw_kernel.group_size()[0]);
  state_->broadcast_fw_kernel.kernel().setArg(0, CDATA(x));
  state_->broadcast_fw_kernel.kernel().setArg(1, skip1);
  state_->broadcast_fw_kernel.kernel().setArg(2, skip2);
  state_->broadcast_fw_kernel.kernel().setArg(3, total);
  state_->broadcast_fw_kernel.kernel().setArg(4, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->broadcast_fw_kernel.kernel(), cl::NullRange,
      cl::NDRange(g1 * state_->broadcast_fw_kernel.group_size()[0]),
      cl::NDRange(state_->broadcast_fw_kernel.group_size()[0]));
}

}  // namespace devices
}  // namespace primitiv
