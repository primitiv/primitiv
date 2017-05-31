#ifndef PRIMITIV_CUDA_DEVICE_H_
#define PRIMITIV_CUDA_DEVICE_H_

#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <curand.h>
#include <map>
#include <primitiv/device.h>

namespace primitiv {

/**
 * Device class for CUDA.
 */
class CUDADevice : public Device {
  CUDADevice() = delete;
  CUDADevice(const CUDADevice &) = delete;
  CUDADevice(CUDADevice &&) = delete;
  CUDADevice &operator=(const CUDADevice &) = delete;
  CUDADevice &operator=(CUDADevice &&) = delete;

public:
  /**
   * Creates a new CUDA device.
   * @param device_id ID of the physical GPU.
   * @remarks The random number generator is initialized using
   *          `std::random_device`.
   */
  explicit CUDADevice(const unsigned device_id);

  /**
   * Creates a new CUDA device.
   * @param device_id ID of the physical GPU.
   * @param rng_seed The seed value of the random number generator.
   */
  CUDADevice(const unsigned device_id, const unsigned rng_seed);

  ~CUDADevice() override;

  using Device::new_tensor;
  Tensor new_tensor(const Shape &shape) override;
  Tensor new_tensor(const Shape &shape, const float k) override;
  Tensor new_tensor(
      const Shape &shape, const std::vector<float> &values) override;
  void delete_tensor(Tensor &x) override;
  std::vector<float> tensor_to_vector(const Tensor &x) override;
  void reset_tensor(Tensor &x, const float k) override;
  void reset_tensor(Tensor &x, const std::vector<float> &values) override;

  Tensor random_bernoulli(const Shape &shape, const float p) override;
  Tensor random_uniform(
      const Shape &shape, const float lower, const float upper) override;
  Tensor random_normal(
      const Shape &shape, const float mean, const float sd) override;

  Tensor slice(
      const Tensor &x, const unsigned dim,
      const unsigned lower, const unsigned upper) override;

  Tensor duplicate(const Tensor &x) override;
  Tensor negate(const Tensor &x) override;

  Tensor add(const Tensor &x, const float k) override;
  Tensor add(const Tensor &a, const Tensor &b) override;
  Tensor subtract(const Tensor &x, const float k) override;
  Tensor subtract(const float k, const Tensor &x) override;
  Tensor subtract(const Tensor &a, const Tensor &b) override;
  Tensor multiply(const Tensor &x, const float k) override;
  Tensor multiply(const Tensor &a, const Tensor &b) override;
  Tensor divide(const Tensor &x, const float k) override;
  Tensor divide(const float k, const Tensor &x) override;
  Tensor divide(const Tensor &a, const Tensor &b) override;

  Tensor transpose(const Tensor &x) override;
  Tensor dot(const Tensor &a, const Tensor &b) override;

  Tensor exp(const Tensor &x) override;
  Tensor tanh(const Tensor &x) override;
  Tensor sigmoid(const Tensor &x) override;
  Tensor step(const Tensor &x) override;
  Tensor relu(const Tensor &x) override;

  Tensor batch_sum(const Tensor &x) override;

  void add_gradient(Tensor &a, const Tensor &b) override;

private:
  unsigned dev_id_;
  unsigned rng_seed_;
  unsigned dim1_x_;
  unsigned dim2_x_;
  unsigned dim2_y_;
  std::map<void *, unsigned> blocks_;
  ::cudaDeviceProp prop_;
  ::cublasHandle_t cublas_;
  ::curandGenerator_t curand_;

  /**
   * Internal method to initialize the object.
   */
  void initialize();
};

}  // namespace primitiv

#endif  // PRIMITIV_CUDA_DEVICE_H_
