#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

template<std::uint32_t BLOCK_SIZE>
__global__ void argmax_dev(
    const float *px, std::uint32_t skip, std::uint32_t n, std::uint32_t *py) {
  __shared__ float max_val[BLOCK_SIZE];
  __shared__ std::uint32_t argmax_val[BLOCK_SIZE];
  const std::uint32_t bid = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  px += bid % skip + (bid / skip) * skip * n;
  max_val[tid] = FLOAT_NEGATIVE_INFINITY;
  for (std::uint32_t i = tid; i < n; i += BLOCK_SIZE) {
    const float val = px[i * skip];
    if (val > max_val[tid]) {
      max_val[tid] = val;
      argmax_val[tid] = i;
    }
  }
  ::__syncthreads();
#define REDUCE(k) \
  if (BLOCK_SIZE >= k << 1) { \
    if (tid < k) { \
      if (max_val[tid + k] > max_val[tid] \
          || (max_val[tid + k] == max_val[tid] \
              && argmax_val[tid + k] < argmax_val[tid])) { \
        max_val[tid] = max_val[tid + k]; \
        argmax_val[tid] = argmax_val[tid + k]; \
      } \
    } \
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
  if (tid == 0) py[bid] = argmax_val[0];
}

}  // namespace

namespace primitiv {
namespace devices {

std::vector<std::uint32_t> CUDA::argmax_impl(
    const Tensor &x, std::uint32_t dim) {
  const Shape &shape = x.shape();
  const std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  const std::uint32_t s = shape.lower_volume(dim);
  std::uint32_t block_size = dim1_x_;
  while (block_size >> 1 >= n) block_size >>= 1;
  std::shared_ptr<void> py = state_->pool.allocate(sizeof(std::uint32_t) * r);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  switch (block_size) {
#define CASE(k) \
    case k: ::argmax_dev<k><<<r, k>>>( \
        CDATA(x), s, n, static_cast<std::uint32_t *>(py.get())); break
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
  std::vector<std::uint32_t> ret(r);
  CUDA_CALL(::cudaMemcpy(
        &ret[0], py.get(), sizeof(std::uint32_t) * r, cudaMemcpyDeviceToHost));
  return ret;
}

}  // namespace devices
}  // namespace primitiv
