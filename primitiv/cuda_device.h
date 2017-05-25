#ifndef PRIMITIV_CUDA_DEVICE_H_
#define PRIMITIV_CUDA_DEVICE_H_

#include <cuda_runtime_api.h>
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
   */
  explicit CUDADevice(const unsigned device_id);
  
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

  void add_gradient(Tensor &a, const Tensor &b) override;

private:
  unsigned dev_id_;
  ::cudaDeviceProp prop_;
  std::map<void *, unsigned> blocks_;
};

}  // namespace primitiv

#endif  // PRIMITIV_CUDA_DEVICE_H_
