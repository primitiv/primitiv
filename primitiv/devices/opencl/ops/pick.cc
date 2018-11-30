#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::pick_fw_impl(
    const Tensor &x, const std::vector<std::uint32_t> &ids,
    std::uint32_t dim, Tensor &y) {
  if (!state_->pick_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->pick_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/pick.inc"
    }), "pick_fw_kernel");
  }
  const std::uint32_t wy = y.shape().lower_volume(dim);
  const std::uint32_t wx = wy * x.shape()[dim];
  const std::uint32_t sx = x.shape().has_batch() * x.shape().volume();
  const std::uint32_t si = ids.size() > 1;
  const std::uint32_t sy = y.shape().volume();
  const std::uint32_t g1 = ::calc_num_blocks(sy, state_->pick_fw_kernel.group_size()[0]);
  const std::uint32_t bs = y.shape().batch();
  std::shared_ptr<void> ids_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * ids.size());
  ::write_buffer(state_->queue, ::get_buffer(ids_buf), ids.data(), ids.size());
  state_->pick_fw_kernel.kernel().setArg(0, CDATA(x));
  state_->pick_fw_kernel.kernel().setArg(1, ::get_buffer(ids_buf));
  state_->pick_fw_kernel.kernel().setArg(2, wx);
  state_->pick_fw_kernel.kernel().setArg(3, wy);
  state_->pick_fw_kernel.kernel().setArg(4, sx);
  state_->pick_fw_kernel.kernel().setArg(5, si);
  state_->pick_fw_kernel.kernel().setArg(6, sy);
  state_->pick_fw_kernel.kernel().setArg(7, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->pick_fw_kernel.kernel(), cl::NullRange,
      cl::NDRange(g1 * state_->pick_fw_kernel.group_size()[0], bs),
      cl::NDRange(state_->pick_fw_kernel.group_size()[0], 1));
}

void OpenCL::pick_bw_impl(
    const Tensor &gy, const std::vector<std::uint32_t> &ids,
    std::uint32_t dim, Tensor &gx) {
  if (!state_->pick_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->pick_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/pick.inc"
    }), "pick_bw_kernel");
  }
  const std::uint32_t wy = gy.shape().lower_volume(dim);
  const std::uint32_t wx = wy * gx.shape()[dim];
  const std::uint32_t sx = gx.shape().has_batch() * gx.shape().volume();
  const std::uint32_t si = ids.size() > 1;
  const std::uint32_t sy = gy.shape().volume();
  const std::uint32_t g1 = ::calc_num_blocks(sy, state_->pick_bw_kernel.group_size()[0]);
  const std::uint32_t bs = gy.shape().batch();
  std::shared_ptr<void> ids_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * ids.size());
  ::write_buffer(state_->queue, ::get_buffer(ids_buf), ids.data(), ids.size());
  state_->pick_bw_kernel.kernel().setArg(0, CDATA(gy));
  state_->pick_bw_kernel.kernel().setArg(1, ::get_buffer(ids_buf));
  state_->pick_bw_kernel.kernel().setArg(2, wx);
  state_->pick_bw_kernel.kernel().setArg(3, wy);
  state_->pick_bw_kernel.kernel().setArg(4, sx);
  state_->pick_bw_kernel.kernel().setArg(5, si);
  state_->pick_bw_kernel.kernel().setArg(6, sy);
  state_->pick_bw_kernel.kernel().setArg(7, MDATA(gx));
  state_->queue.enqueueNDRangeKernel(
      state_->pick_bw_kernel.kernel(), cl::NullRange,
      cl::NDRange(g1 * state_->pick_bw_kernel.group_size()[0], bs),
      cl::NDRange(state_->pick_bw_kernel.group_size()[0], 1));
}

}  // namespace devices
}  // namespace primitiv
