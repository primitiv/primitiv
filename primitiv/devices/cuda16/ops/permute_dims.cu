#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void permute_dims_fw_dev(
    const half *px, const std::uint32_t ndims, const std::uint32_t *x_strides,
    const std::uint32_t *y_strides, const std::uint32_t size, half *py) {
  const std::uint32_t i = IDX;
  const std::uint32_t bid_z = IDY;
  const std::uint32_t ofs = bid_z * size;
  if (i < size) {
    std::uint32_t tmp = i;
    std::uint32_t j = 0;
    for (std::uint32_t d = 0; d < ndims; ++d) {
      const std::uint32_t p = tmp / x_strides[d];
      tmp -= p * x_strides[d];
      j += p * y_strides[d];
    }
    py[ofs + j] = px[ofs + i];
  }
}

__global__ void permute_dims_bw_dev(
    const half *py, const std::uint32_t ndims, const std::uint32_t *x_strides,
    const std::uint32_t *y_strides, const std::uint32_t size, half *px) {
  const std::uint32_t i = IDX;
  const std::uint32_t bid_z = IDY;
  const std::uint32_t ofs = bid_z * size;
  if (i < size) {
    std::uint32_t tmp = i;
    std::uint32_t j = 0;
    for (std::uint32_t d = 0; d < ndims; ++d) {
      const std::uint32_t p = tmp / x_strides[d];
      tmp -= p * x_strides[d];
      j += p * y_strides[d];
    }
    const std::size_t ox = ofs + i;
    const std::size_t oy = ofs + j;
    INPLACE_ADD(px + ox, ::__half2float(py[oy]));
  }
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::permute_dims_fw_impl(
    const Tensor &x, const std::vector<std::uint32_t> &perm,
    Tensor &y) {
  const std::uint32_t ndims = perm.size();
  const std::uint32_t bs = x.shape().batch();
  const std::uint32_t size = x.shape().volume();
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_);
  std::vector<std::uint32_t> x_strides(ndims);
  std::vector<std::uint32_t> y_strides(ndims);
  for (std::uint32_t i = 0; i < ndims; ++i) {
    x_strides[ndims - i - 1] = x.shape().lower_volume(i);
    y_strides[ndims - perm[i] - 1] = y.shape().lower_volume(i);
  }
  std::shared_ptr<void> x_strides_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * x_strides.size());
  std::shared_ptr<void> y_strides_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * y_strides.size());
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        x_strides_buf.get(), x_strides.data(), sizeof(std::uint32_t) * x_strides.size(),
        cudaMemcpyHostToDevice));
  CUDA_CALL(::cudaMemcpy(
        y_strides_buf.get(), y_strides.data(), sizeof(std::uint32_t) * y_strides.size(),
        cudaMemcpyHostToDevice));
  ::permute_dims_fw_dev<<<dim3(g1, bs), dim1_x_>>>(
      CDATA(half, x), ndims,
      static_cast<const std::uint32_t *>(x_strides_buf.get()),
      static_cast<const std::uint32_t *>(y_strides_buf.get()),
      size, MDATA(half, y));
}

void CUDA16::permute_dims_bw_impl(
    const Tensor &, const Tensor &, const Tensor &gy,
    const std::vector<std::uint32_t> &perm, Tensor &gx) {
  const std::uint32_t ndims = perm.size();
  const std::uint32_t bs = gx.shape().batch();
  const std::uint32_t size = gx.shape().volume();
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_);
  std::vector<std::uint32_t> x_strides(ndims);
  std::vector<std::uint32_t> y_strides(ndims);
  for (std::uint32_t i = 0; i < ndims; ++i) {
    x_strides[ndims - i - 1] = gx.shape().lower_volume(i);
    y_strides[ndims - perm[i] - 1] = gy.shape().lower_volume(i);
  }
  std::shared_ptr<void> x_strides_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * x_strides.size());
  std::shared_ptr<void> y_strides_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * y_strides.size());
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        x_strides_buf.get(), x_strides.data(), sizeof(std::uint32_t) * x_strides.size(),
        cudaMemcpyHostToDevice));
  CUDA_CALL(::cudaMemcpy(
        y_strides_buf.get(), y_strides.data(), sizeof(std::uint32_t) * y_strides.size(),
        cudaMemcpyHostToDevice));
  ::permute_dims_bw_dev<<<dim3(g1, bs), dim1_x_>>>(
      CDATA(half, gy), ndims,
      static_cast<const std::uint32_t *>(x_strides_buf.get()),
      static_cast<const std::uint32_t *>(y_strides_buf.get()),
      size, MDATA(half, gx));
}

}  // namespace devices
}  // namespace primitiv
