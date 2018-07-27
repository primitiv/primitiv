#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

template<std::uint32_t BLOCK_SIZE>
__global__ void max_fw_dev(
    const half *px, std::uint32_t skip, std::uint32_t n, half *py) {
  __shared__ float temp[BLOCK_SIZE];
  const std::uint32_t bid = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  px += bid % skip + (bid / skip) * skip * n;
  float thread_max = FLOAT_NEGATIVE_INFINITY;
  for (std::uint32_t i = tid; i < n; i += BLOCK_SIZE) {
    thread_max = fmaxf(::__half2float(px[i * skip]), thread_max);
  }
  temp[tid] = thread_max;
  ::__syncthreads();
#define REDUCE(k) \
  if (BLOCK_SIZE >= k << 1) { \
    if (tid < k) temp[tid] = fmaxf(temp[tid + k], temp[tid]); \
    ::__syncthreads(); \
  }
  REDUCE(512)
  REDUCE(256)
  REDUCE(128)
  REDUCE(64)
  REDUCE(32)
  REDUCE(16)
  REDUCE(8)
  REDUCE(4)
  REDUCE(2)
  REDUCE(1)
#undef REDUCE
  if (tid == 0) py[bid] = ::__float2half(temp[0]);
}

template<std::uint32_t BLOCK_SIZE>
__global__ void max_bw_dev(
    const half *px, const half *py, const half *pgy,
    std::uint32_t skip, std::uint32_t n, half *pgx) {
  __shared__ std::uint32_t argmax_val[BLOCK_SIZE];
  const std::uint32_t bid = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  const float max_val = ::__half2float(py[bid]);
  px += bid % skip + (bid / skip) * skip * n;
  pgx += bid % skip + (bid / skip) * skip * n;
  std::uint32_t thread_argmin = n;
  for (std::uint32_t i = tid; i < n; i += BLOCK_SIZE) {
    if (::__half2float(px[i * skip]) == max_val) {
      thread_argmin = min(i, thread_argmin);
    }
  }
  argmax_val[tid] = thread_argmin;
  ::__syncthreads();
#define REDUCE(k) \
  if (BLOCK_SIZE >= k << 1) { \
    if (tid < k) argmax_val[tid] = min(argmax_val[tid + k], argmax_val[tid]); \
    ::__syncthreads(); \
  }
  REDUCE(512)
  REDUCE(256)
  REDUCE(128)
  REDUCE(64)
  REDUCE(32)
  REDUCE(16)
  REDUCE(8)
  REDUCE(4)
  REDUCE(2)
  REDUCE(1)
#undef REDUCE
  if (tid == 0) INPLACE_ADD(pgx + argmax_val[0] * skip, ::__half2float(pgy[bid]));
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::max_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t block_size = dim1_x_;
  while (block_size >> 1 >= n) block_size >>= 1;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  switch (block_size) {
#define CASE(k) \
    case k: \
      ::max_fw_dev<k><<<r, k>>>(CDATA(half, x), s, n, MDATA(half, y)); \
      break;
    CASE(1024);
    CASE(512);
    CASE(256);
    CASE(128);
    CASE(64);
    CASE(32);
    CASE(16);
    CASE(8);
    CASE(4);
    CASE(2);
    CASE(1);
#undef CASE
  }
}

void CUDA16::max_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy,
    std::uint32_t dim, Tensor &gx) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t block_size = dim1_x_;
  while (block_size >> 1 >= n) block_size >>= 1;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  switch (block_size) {
#define CASE(k) \
    case k: \
      ::max_bw_dev<k><<<r, k>>>( \
          CDATA(half, x), CDATA(half, y), CDATA(half, gy), \
          s, n, MDATA(half, gx)); \
      break;
    CASE(1024);
    CASE(512);
    CASE(256);
    CASE(128);
    CASE(64);
    CASE(32);
    CASE(16);
    CASE(8);
    CASE(4);
    CASE(2);
    CASE(1);
#undef CASE
  }
}

}  // namespace devices
}  // namespace primitiv
