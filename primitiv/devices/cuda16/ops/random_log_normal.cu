#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void fp32to16(const float *px, std::size_t size, half *py) {
  const std::size_t i = IDX;
  if (i < size) py[i] = ::__float2half(px[i]);
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::random_log_normal_impl(float mean, float sd, Tensor &y) {
  const std::size_t size = y.shape().size();
  const std::size_t gs = GRID_SIZE(size, dim1_x_);
  std::size_t allocated_size;
  auto temp = state_->pool.allocate(size * sizeof(float), &allocated_size);
  float *temp_ptr = static_cast<float *>(temp.get());

  std::size_t n = size;
  if (n % 2 != 0) {
    std::size_t capacity = allocated_size / sizeof(float);
    if (capacity <= n) {
      PRIMITIV_THROW_ERROR(
          "Could not generate " << n
          << " + 1 random values. capacity: " << capacity);
    }
    ++n;
  }
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateLogNormal(
        state_->curand.get(), temp_ptr, n, mean, sd));
  ::fp32to16<<<gs, dim1_x_>>>(temp_ptr, size, MDATA(half, y));
}

}  // namespace devices
}  // namespace primitiv
