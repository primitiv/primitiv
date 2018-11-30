#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::flip_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  if (!state_->flip_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->flip_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/flip.inc"
    }), "flip_fw_kernel");
  }
  const Shape &s = x.shape();
  const std::uint32_t n = s[dim];
  const std::uint32_t skip = s.lower_volume(dim);
  const std::uint32_t r = s.size() / n;
  const std::uint32_t g1 = ::calc_num_blocks(
      n, state_->flip_fw_kernel.group_size()[0]);
  const std::uint32_t g2 = ::calc_num_blocks(
      r, state_->flip_fw_kernel.group_size()[1]);
  state_->flip_fw_kernel.kernel().setArg(0, CDATA(x));
  state_->flip_fw_kernel.kernel().setArg(1, skip);
  state_->flip_fw_kernel.kernel().setArg(2, n);
  state_->flip_fw_kernel.kernel().setArg(3, r);
  state_->flip_fw_kernel.kernel().setArg(4, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->flip_fw_kernel.kernel(), cl::NullRange,
      cl::NDRange(
          g1 * state_->flip_fw_kernel.group_size()[0],
          g2 * state_->flip_fw_kernel.group_size()[1]),
      cl::NDRange(
          state_->flip_fw_kernel.group_size()[0],
          state_->flip_fw_kernel.group_size()[1]));
}

void OpenCL::flip_bw_impl(const Tensor &gy, std::uint32_t dim, Tensor &gx) {
  if (!state_->flip_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->flip_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/flip.inc"
    }), "flip_bw_kernel");
  }
  const Shape &s = gy.shape();
  const std::uint32_t n = s[dim];
  const std::uint32_t skip = s.lower_volume(dim);
  const std::uint32_t r = s.size() / n;
  const std::uint32_t g1 = ::calc_num_blocks(
      n, state_->flip_bw_kernel.group_size()[0]);
  const std::uint32_t g2 = ::calc_num_blocks(
      r, state_->flip_bw_kernel.group_size()[1]);
  state_->flip_bw_kernel.kernel().setArg(0, CDATA(gy));
  state_->flip_bw_kernel.kernel().setArg(1, skip);
  state_->flip_bw_kernel.kernel().setArg(2, n);
  state_->flip_bw_kernel.kernel().setArg(3, r);
  state_->flip_bw_kernel.kernel().setArg(4, MDATA(gx));
  state_->queue.enqueueNDRangeKernel(
      state_->flip_bw_kernel.kernel(), cl::NullRange,
      cl::NDRange(
          g1 * state_->flip_bw_kernel.group_size()[0],
          g2 * state_->flip_bw_kernel.group_size()[1]),
      cl::NDRange(
          state_->flip_bw_kernel.group_size()[0],
          state_->flip_bw_kernel.group_size()[1]));
}

}  // namespace devices
}  // namespace primitiv
