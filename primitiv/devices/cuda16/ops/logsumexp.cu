#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__device__ float logsumexp2_fw_dev(float a, float b) {
  return a > b
    ? a + ::log(1.f + ::exp(b - a))
    : b + ::log(1.f + ::exp(a - b));
}

template<std::uint32_t BLOCK_SIZE>
__global__ void logsumexp_fw_dev(
    const half *px, std::uint32_t skip, std::uint32_t n, half *py) {
  __shared__ float temp[BLOCK_SIZE];
  const std::uint32_t bid = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  px += bid % skip + (bid / skip) * skip * n;
  temp[tid] = FLOAT_NEGATIVE_INFINITY;
  for (std::uint32_t i = tid; i < n; i += BLOCK_SIZE) {
    temp[tid] = ::logsumexp2_fw_dev(temp[tid], ::__half2float(px[i * skip]));
  }
  ::__syncthreads();
#define REDUCE(k) \
  if (BLOCK_SIZE >= k << 1) { \
    if (tid < k) temp[tid] = ::logsumexp2_fw_dev(temp[tid], temp[tid + k]); \
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

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::logsumexp_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t block_size = dim1_x_;
  while (block_size >> 1 >= n) block_size >>= 1;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  switch (block_size) {
#define CASE(k) \
    case k: \
      ::logsumexp_fw_dev<k><<<r, k>>>(CDATA(half, x), s, n, MDATA(half, y)); \
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
