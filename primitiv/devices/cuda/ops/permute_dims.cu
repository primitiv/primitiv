#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__constant__ std::uint32_t permute_dims_x_strides[primitiv::Shape::MAX_DEPTH];
__constant__ std::uint32_t permute_dims_y_strides[primitiv::Shape::MAX_DEPTH];

// TODO(vbkaisetsu):
// Implove implementation of permute_dims.
// This function uses for-loops in the kernel code. It becomes slower than
// no-loop implementation.
__global__ void permute_dims_fw_dev(
    const float *px, const std::uint32_t ndims, const std::uint32_t size,
    float *py) {
  const std::uint32_t i = IDX;
  const std::uint32_t bid_z = IDY;
  const std::uint32_t ofs = bid_z * size;
  if (i < size) {
    std::uint32_t tmp = i;
    std::uint32_t j = 0;
    // TODO(vbkaisetsu):
    // Implove implementation
    for (std::uint32_t d = 0; d < ndims; ++d) {
      const std::uint32_t p = tmp / permute_dims_x_strides[d];
      tmp -= p * permute_dims_x_strides[d];
      j += p * permute_dims_y_strides[d];
    }
    py[ofs + j] = px[ofs + i];
  }
}

__global__ void permute_dims_bw_dev(
    const float *py, const std::uint32_t ndims, const std::uint32_t size,
    float *px) {
  const std::uint32_t i = IDX;
  const std::uint32_t bid_z = IDY;
  const std::uint32_t ofs = bid_z * size;
  if (i < size) {
    std::uint32_t tmp = i;
    std::uint32_t j = 0;
    // TODO(vbkaisetsu):
    // Implove implementation
    for (std::uint32_t d = 0; d < ndims; ++d) {
      const std::uint32_t p = tmp / permute_dims_x_strides[d];
      tmp -= p * permute_dims_x_strides[d];
      j += p * permute_dims_y_strides[d];
    }
    px[ofs + i] += py[ofs + j];
  }
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA::permute_dims_fw_impl(
    const Tensor &x, const std::vector<std::uint32_t> &perm,
    Tensor &y) {
  const std::uint32_t ndims = perm.size();
  const std::uint32_t bs = x.shape().batch();
  const std::uint32_t size = x.shape().volume();
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_);
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
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpyToSymbol(
      permute_dims_x_strides, x_strides.data(),
      sizeof(std::uint32_t) * x_strides.size()));
  CUDA_CALL(::cudaMemcpyToSymbol(
      permute_dims_y_strides, y_strides.data(),
      sizeof(std::uint32_t) * y_strides.size()));
  ::permute_dims_fw_dev<<<dim3(g1, bs), dim1_x_>>>(
      CDATA(x), ndims, size, MDATA(y));
}

void CUDA::permute_dims_bw_impl(
    const Tensor &, const Tensor &, const Tensor &gy,
    const std::vector<std::uint32_t> &perm, Tensor &gx) {
  const std::uint32_t ndims = perm.size();
  const std::uint32_t bs = gx.shape().batch();
  const std::uint32_t size = gx.shape().volume();
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_);
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
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpyToSymbol(
      permute_dims_x_strides, x_strides.data(),
      sizeof(std::uint32_t) * x_strides.size()));
  CUDA_CALL(::cudaMemcpyToSymbol(
      permute_dims_y_strides, y_strides.data(),
      sizeof(std::uint32_t) * y_strides.size()));
  ::permute_dims_bw_dev<<<dim3(g1, bs), dim1_x_>>>(
      CDATA(gy), ndims, size, MDATA(gx));
}

}  // namespace devices
}  // namespace primitiv
