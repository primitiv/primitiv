#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::flip_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const Shape &s = x.shape();
  const std::uint32_t n = s[dim];
  const std::uint32_t skip = s.lower_volume(dim);
  const std::uint32_t r = s.size() / n;
  const std::uint32_t g1 = ::calc_num_blocks(
      n, state_->flip_fw_group_size_x);
  const std::uint32_t g2 = ::calc_num_blocks(
      r, state_->flip_fw_group_size_y);
  state_->flip_fw_kernel.setArg(0, CDATA(x));
  state_->flip_fw_kernel.setArg(1, skip);
  state_->flip_fw_kernel.setArg(2, n);
  state_->flip_fw_kernel.setArg(3, r);
  state_->flip_fw_kernel.setArg(4, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->flip_fw_kernel, cl::NullRange,
      cl::NDRange(
          g1 * state_->flip_fw_group_size_x,
          g2 * state_->flip_fw_group_size_y),
      cl::NDRange(
          state_->flip_fw_group_size_x,
          state_->flip_fw_group_size_y));
}

void OpenCL::flip_bw_impl(const Tensor &gy, std::uint32_t dim, Tensor &gx) {
  const Shape &s = gy.shape();
  const std::uint32_t n = s[dim];
  const std::uint32_t skip = s.lower_volume(dim);
  const std::uint32_t r = s.size() / n;
  const std::uint32_t g1 = ::calc_num_blocks(
      n, state_->flip_bw_group_size_x);
  const std::uint32_t g2 = ::calc_num_blocks(
      r, state_->flip_bw_group_size_y);
  state_->flip_bw_kernel.setArg(0, CDATA(gy));
  state_->flip_bw_kernel.setArg(1, skip);
  state_->flip_bw_kernel.setArg(2, n);
  state_->flip_bw_kernel.setArg(3, r);
  state_->flip_bw_kernel.setArg(4, MDATA(gx));
  state_->queue.enqueueNDRangeKernel(
      state_->flip_bw_kernel, cl::NullRange,
      cl::NDRange(
          g1 * state_->flip_bw_group_size_x,
          g2 * state_->flip_bw_group_size_y),
      cl::NDRange(
          state_->flip_bw_group_size_x,
          state_->flip_bw_group_size_y));
}

}  // namespace devices
}  // namespace primitiv
