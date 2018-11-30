#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::slice_fw_impl(
    const Tensor &x, std::uint32_t dim, std::uint32_t offset, Tensor &y) {
  if (!state_->slice_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->slice_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/slice.inc"
    }), "slice_fw_kernel");
  }
  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t shift = base * offset;
  const std::uint32_t span = base * y.shape()[dim];
  const std::uint32_t skip = base * x.shape()[dim];
  const std::uint32_t size = y.shape().size();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->slice_fw_kernel.group_size()[0]);
  state_->slice_fw_kernel.kernel().setArg(0, CDATA(x));
  state_->slice_fw_kernel.kernel().setArg(1, shift);
  state_->slice_fw_kernel.kernel().setArg(2, span);
  state_->slice_fw_kernel.kernel().setArg(3, skip);
  state_->slice_fw_kernel.kernel().setArg(4, size);
  state_->slice_fw_kernel.kernel().setArg(5, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->slice_fw_kernel.kernel(), cl::NullRange,
      cl::NDRange(num_blocks * state_->slice_fw_kernel.group_size()[0]),
      cl::NDRange(state_->slice_fw_kernel.group_size()[0]));
}

void OpenCL::slice_bw_impl(
    const Tensor &gy, std::uint32_t dim, std::uint32_t offset, Tensor &gx) {
  if (!state_->slice_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->slice_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/slice.inc"
    }), "slice_bw_kernel");
  }
  const Shape &sx = gx.shape();
  const Shape &sy = gy.shape();
  const std::uint32_t base = sx.lower_volume(dim);
  const std::uint32_t ox = base * offset;
  const std::uint32_t wx = base * sx[dim];
  const std::uint32_t wy = base * sy[dim];
  const std::uint32_t repeat = sx.volume() / wx;
  const std::uint32_t nx = repeat * sx.batch();
  const std::uint32_t ny = repeat * sy.batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      wy * std::max(nx, ny), state_->slice_bw_kernel.group_size()[0]);
  state_->slice_bw_kernel.kernel().setArg(0, CDATA(gy));
  state_->slice_bw_kernel.kernel().setArg(1, wx);
  state_->slice_bw_kernel.kernel().setArg(2, wy);
  state_->slice_bw_kernel.kernel().setArg(3, nx);
  state_->slice_bw_kernel.kernel().setArg(4, ny);
  state_->slice_bw_kernel.kernel().setArg(5, MDATA(gx));
  state_->slice_bw_kernel.kernel().setArg(6, ox);
  state_->queue.enqueueNDRangeKernel(
      state_->slice_bw_kernel.kernel(), cl::NullRange,
      cl::NDRange(g1 * state_->slice_bw_kernel.group_size()[0]),
      cl::NDRange(state_->slice_bw_kernel.group_size()[0]));
}

}  // namespace devices
}  // namespace primitiv
