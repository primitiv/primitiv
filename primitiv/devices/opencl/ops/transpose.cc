#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::transpose_fw_impl(const Tensor &x, Tensor &y) {
  if (!state_->transpose_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->transpose_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/transpose.inc"
    }), "transpose_fw_kernel");
  }
  const std::uint32_t rows = x.shape()[0];
  const std::uint32_t cols = x.shape()[1];
  const std::uint32_t bs = x.shape().batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      rows, state_->transpose_fw_kernel.group_size()[0]);
  const std::uint32_t g2 = ::calc_num_blocks(
      cols, state_->transpose_fw_kernel.group_size()[1]);
  state_->transpose_fw_kernel.kernel().setArg(0, CDATA(x));
  state_->transpose_fw_kernel.kernel().setArg(1, rows);
  state_->transpose_fw_kernel.kernel().setArg(2, cols);
  state_->transpose_fw_kernel.kernel().setArg(3, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->transpose_fw_kernel.kernel(), cl::NullRange,
      cl::NDRange(
        g1 * state_->transpose_fw_kernel.group_size()[0],
        g2 * state_->transpose_fw_kernel.group_size()[1], bs),
      cl::NDRange(
        state_->transpose_fw_kernel.group_size()[0],
        state_->transpose_fw_kernel.group_size()[1], 1));
}

void OpenCL::transpose_bw_impl(
    const Tensor &, const Tensor &, const Tensor &gy, Tensor &gx) {
  if (!state_->transpose_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->transpose_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/transpose.inc"
    }), "transpose_bw_kernel");
  }
  const std::uint32_t rows = gx.shape()[0];
  const std::uint32_t cols = gx.shape()[1];
  const std::uint32_t bs = gx.shape().batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      rows, state_->transpose_bw_kernel.group_size()[0]);
  const std::uint32_t g2 = ::calc_num_blocks(
      cols, state_->transpose_bw_kernel.group_size()[1]);
  state_->transpose_bw_kernel.kernel().setArg(0, CDATA(gy));
  state_->transpose_bw_kernel.kernel().setArg(1, rows);
  state_->transpose_bw_kernel.kernel().setArg(2, cols);
  state_->transpose_bw_kernel.kernel().setArg(3, MDATA(gx));
  state_->queue.enqueueNDRangeKernel(
      state_->transpose_bw_kernel.kernel(), cl::NullRange,
      cl::NDRange(
        g1 * state_->transpose_bw_kernel.group_size()[0],
        g2 * state_->transpose_bw_kernel.group_size()[1], bs),
      cl::NDRange(
        state_->transpose_bw_kernel.group_size()[0],
        state_->transpose_bw_kernel.group_size()[1], 1));
}

}  // namespace devices
}  // namespace primitiv
