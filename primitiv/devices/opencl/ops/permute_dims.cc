#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

// TODO(vbkaisetsu):
// Implove implementation of permute_dims.
// This function uses for-loops in the kernel code. It becomes slower than
// no-loop implementation.
void OpenCL::permute_dims_fw_impl(
    const Tensor &x, const std::vector<std::uint32_t> &perm,
    Tensor &y) {
  const std::uint32_t ndims = perm.size();
  const std::uint32_t bs = x.shape().batch();
  const std::uint32_t size = x.shape().volume();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->permute_dims_fw_group_size);
  std::vector<std::uint32_t> x_strides(ndims);
  std::vector<std::uint32_t> y_strides(ndims);
  std::uint32_t x_stride_tmp = 1;
  std::uint32_t y_stride_tmp = 1;
  for (std::uint32_t i = 0; i < ndims; ++i) {
    x_strides[ndims - i - 1] = x_stride_tmp;
    y_strides[ndims - perm[i] - 1] = y_stride_tmp;
    x_stride_tmp *= x.shape()[i];
    y_stride_tmp *= y.shape()[i];
  }
  std::shared_ptr<void> x_strides_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * x_strides.size());
  std::shared_ptr<void> y_strides_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * y_strides.size());
  if (perm.size() != 0) {
    ::write_buffer(state_->queue, ::get_buffer(x_strides_buf), x_strides.data(), x_strides.size());
    ::write_buffer(state_->queue, ::get_buffer(y_strides_buf), y_strides.data(), y_strides.size());
  }
  state_->permute_dims_fw_kernel.setArg(0, CDATA(x));
  state_->permute_dims_fw_kernel.setArg(1, ndims);
  state_->permute_dims_fw_kernel.setArg(2, ::get_buffer(x_strides_buf));
  state_->permute_dims_fw_kernel.setArg(3, ::get_buffer(y_strides_buf));
  state_->permute_dims_fw_kernel.setArg(4, size);
  state_->permute_dims_fw_kernel.setArg(5, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->permute_dims_fw_kernel, cl::NullRange,
      cl::NDRange(num_blocks * state_->transpose_fw_group_size, bs),
      cl::NDRange(state_->permute_dims_fw_group_size, 1));
}

void OpenCL::permute_dims_bw_impl(
    const Tensor &, const Tensor &, const Tensor &gy,
    const std::vector<std::uint32_t> &perm, Tensor &gx) {
  const std::uint32_t ndims = perm.size();
  const std::uint32_t bs = gx.shape().batch();
  const std::uint32_t size = gx.shape().volume();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->permute_dims_bw_group_size);
  std::vector<std::uint32_t> x_strides(ndims);
  std::vector<std::uint32_t> y_strides(ndims);
  std::uint32_t x_stride_tmp = 1;
  std::uint32_t y_stride_tmp = 1;
  for (std::uint32_t i = 0; i < ndims; ++i) {
    x_strides[ndims - i - 1] = x_stride_tmp;
    y_strides[ndims - perm[i] - 1] = y_stride_tmp;
    x_stride_tmp *= gx.shape()[i];
    y_stride_tmp *= gy.shape()[i];
  }
  std::shared_ptr<void> x_strides_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * x_strides.size());
  std::shared_ptr<void> y_strides_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * y_strides.size());
  if (perm.size() != 0) {
    ::write_buffer(state_->queue, ::get_buffer(x_strides_buf), x_strides.data(), x_strides.size());
    ::write_buffer(state_->queue, ::get_buffer(y_strides_buf), y_strides.data(), y_strides.size());
  }
  state_->permute_dims_bw_kernel.setArg(0, CDATA(gy));
  state_->permute_dims_bw_kernel.setArg(1, ndims);
  state_->permute_dims_bw_kernel.setArg(2, ::get_buffer(x_strides_buf));
  state_->permute_dims_bw_kernel.setArg(3, ::get_buffer(y_strides_buf));
  state_->permute_dims_bw_kernel.setArg(4, size);
  state_->permute_dims_bw_kernel.setArg(5, MDATA(gx));
  state_->queue.enqueueNDRangeKernel(
      state_->permute_dims_bw_kernel, cl::NullRange,
      cl::NDRange(num_blocks * state_->transpose_bw_group_size, bs),
      cl::NDRange(state_->permute_dims_bw_group_size, 1));
}

}  // namespace devices
}  // namespace primitiv
