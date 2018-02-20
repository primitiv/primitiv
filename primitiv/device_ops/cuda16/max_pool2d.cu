#include <primitiv/config.h>

#include <cstring>

#include <primitiv/cuda16_device.h>
#include <primitiv/internal/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace primitiv {
namespace devices {

void CUDA16::max_pool2d_fw_impl(
    const Tensor &x,
    std::uint32_t window0, std::uint32_t window1,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1,
    Tensor &y) {
  const Shape x_shape = x.shape();
  const Shape y_shape = y.shape();

  // Specifies a target device.
  CUDA_CALL(::cudaSetDevice(dev_id_));

  // Prepares descriptors.
  const cuda::CuDNNTensorDescriptor x_desc(
      x_shape.batch(), x_shape[2], x_shape[1], x_shape[0],
      ::CUDNN_DATA_HALF);
  const cuda::CuDNNTensorDescriptor y_desc(
      y_shape.batch(), y_shape[2], y_shape[1], y_shape[0],
      ::CUDNN_DATA_HALF);
  const cuda::CuDNNPoolingDescriptor pool_desc(
      CUDNN_POOLING_MAX,
      window1, window0, padding1, padding0, stride1, stride0);

  // Performs a forward operation.
  const float alpha = 1.f;
  const float beta = 0.f;
  const half *x_ptr = CDATA(half, x);
  half *y_ptr = MDATA(half, y);
  CUDNN_CALL(::cudnnPoolingForward(
        state_->cudnn.get(), pool_desc.get(),
        &alpha, x_desc.get(), x_ptr,
        &beta, y_desc.get(), y_ptr));
}

void CUDA16::max_pool2d_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy,
    std::uint32_t window0, std::uint32_t window1,
    std::uint32_t padding0, std::uint32_t padding1,
    std::uint32_t stride0, std::uint32_t stride1,
    Tensor &gx) {
  const Shape x_shape = x.shape();
  const Shape y_shape = y.shape();

  // Specifies a target device.
  CUDA_CALL(::cudaSetDevice(dev_id_));

  // Prepares descriptors.
  const cuda::CuDNNTensorDescriptor x_desc(
      x_shape.batch(), x_shape[2], x_shape[1], x_shape[0],
      ::CUDNN_DATA_HALF);
  const cuda::CuDNNTensorDescriptor y_desc(
      y_shape.batch(), y_shape[2], y_shape[1], y_shape[0],
      ::CUDNN_DATA_HALF);
  const cuda::CuDNNPoolingDescriptor pool_desc(
      CUDNN_POOLING_MAX,
      window1, window0, padding1, padding0, stride1, stride0);

  // Performs a backward operation.
  const float alpha = 1.f;
  const float beta = 1.f;
  const half *x_ptr = CDATA(half, x);
  const half *y_ptr = CDATA(half, y);
  const half *gy_ptr = CDATA(half, gy);
  half *gx_ptr = MDATA(half, gx);
  CUDNN_CALL(::cudnnPoolingBackward(
        state_->cudnn.get(), pool_desc.get(),
        &alpha, y_desc.get(), y_ptr, y_desc.get(), gy_ptr, x_desc.get(), x_ptr,
        &beta, x_desc.get(), gx_ptr));
}

}  // namespace devices
}  // namespace primitiv
