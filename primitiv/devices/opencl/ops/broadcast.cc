#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::broadcast_fw_impl(
    const Tensor &x, std::uint32_t dim, std::uint32_t size, Tensor &y) {
  const std::uint32_t skip1 = y.shape().lower_volume(dim);
  const std::uint32_t skip2 = skip1 * size;
  const std::uint32_t total = y.shape().size();
  const std::uint32_t g1 = ::calc_num_blocks(
      total, state_->broadcast_fw_group_size);
  state_->broadcast_fw_kernel.setArg(0, CDATA(x));
  state_->broadcast_fw_kernel.setArg(1, skip1);
  state_->broadcast_fw_kernel.setArg(2, skip2);
  state_->broadcast_fw_kernel.setArg(3, total);
  state_->broadcast_fw_kernel.setArg(4, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->broadcast_fw_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->broadcast_fw_group_size),
      cl::NDRange(state_->broadcast_fw_group_size));
}

}  // namespace devices
}  // namespace primitiv
