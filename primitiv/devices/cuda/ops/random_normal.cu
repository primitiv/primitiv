#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace primitiv {
namespace devices {

void CUDA::random_normal_impl(float mean, float sd, Tensor &y) {
  std::size_t size = y.shape().size();
  if (size % 2 != 0) {
    std::size_t capacity = y.allocated_size() / sizeof(float);
    if (capacity <= size) {
      PRIMITIV_THROW_ERROR(
          "Could not generate " << size
          << " + 1 random values. capacity: " << capacity);
    }
    ++size;
  }
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateNormal(
        state_->curand.get(), MDATA(y), size, mean, sd));
}

}  // namespace devices
}  // namespace primitiv
