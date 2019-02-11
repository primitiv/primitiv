#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::batch_pick_fw_impl(const Tensor &x, const std::vector<std::uint32_t> &ids, Tensor &y) {
  const std::uint32_t si = ids.size() > 1;
  const std::uint32_t sy = y.shape().volume();
  const std::uint32_t g1 = ::calc_num_blocks(sy, state_->batch_pick_fw_group_size);
  const std::uint32_t bs = y.shape().batch();
  std::shared_ptr<void> ids_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * ids.size());
  ::write_buffer(state_->queue, ::get_buffer(ids_buf), ids.data(), ids.size());
  state_->batch_pick_fw_kernel.setArg(0, CDATA(x));
  state_->batch_pick_fw_kernel.setArg(1, ::get_buffer(ids_buf));
  state_->batch_pick_fw_kernel.setArg(2, si);
  state_->batch_pick_fw_kernel.setArg(3, sy);
  state_->batch_pick_fw_kernel.setArg(4, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->batch_pick_fw_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->batch_pick_fw_group_size, bs),
      cl::NDRange(state_->batch_pick_fw_group_size, 1));
}

void OpenCL::batch_pick_bw_impl(const Tensor &gy, const std::vector<std::uint32_t> &ids, Tensor &gx) {
  const std::uint32_t si = ids.size() > 1;
  const std::uint32_t sy = gy.shape().volume();
  const std::uint32_t g1 = ::calc_num_blocks(sy, state_->batch_pick_bw_group_size);
  const std::uint32_t bs = gy.shape().batch();
  std::shared_ptr<void> ids_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * ids.size());
  ::write_buffer(state_->queue, ::get_buffer(ids_buf), ids.data(), ids.size());
  state_->batch_pick_bw_kernel.setArg(0, CDATA(gy));
  state_->batch_pick_bw_kernel.setArg(1, ::get_buffer(ids_buf));
  state_->batch_pick_bw_kernel.setArg(2, si);
  state_->batch_pick_bw_kernel.setArg(3, sy);
  state_->batch_pick_bw_kernel.setArg(4, MDATA(gx));
  state_->queue.enqueueNDRangeKernel(
      state_->batch_pick_bw_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->batch_pick_bw_group_size, bs),
      cl::NDRange(state_->batch_pick_bw_group_size, 1));
}

}  // namespace devices
}  // namespace primitiv
