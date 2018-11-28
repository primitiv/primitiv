#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::concat_fw_impl(
    const std::vector<const Tensor *> &xs, std::uint32_t dim, Tensor &y) {
  const std::uint32_t new_bs = y.shape().batch();
  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t skip = base * y.shape()[dim];
  const std::uint32_t repeat = y.shape().volume() / skip;
  std::uint32_t offset = 0;
  for (const Tensor *x : xs) {
    const std::uint32_t span = base * x->shape()[dim];
    const std::uint32_t x_size = span * repeat * x->shape().batch();
    const std::uint32_t y_size = span * repeat * new_bs;
    const std::uint32_t num_blocks = ::calc_num_blocks(
        y_size, state_->concat_fw_group_size);
    state_->concat_fw_kernel.setArg(0, CDATA(*x));
    state_->concat_fw_kernel.setArg(1, span);
    state_->concat_fw_kernel.setArg(2, skip);
    state_->concat_fw_kernel.setArg(3, x_size);
    state_->concat_fw_kernel.setArg(4, y_size);
    state_->concat_fw_kernel.setArg(5, MDATA(y));
    state_->concat_fw_kernel.setArg(6, offset);
    state_->queue.enqueueNDRangeKernel(
        state_->concat_fw_kernel, cl::NullRange,
        cl::NDRange(num_blocks * state_->concat_fw_group_size),
        cl::NDRange(state_->concat_fw_group_size), nullptr, nullptr);
    offset += span;
  }
}

}  // namespace devices
}  // namespace primitiv
