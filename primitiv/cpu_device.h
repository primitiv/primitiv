#ifndef PRIMITIV_CPU_DEVICE_H_
#define PRIMITIV_CPU_DEVICE_H_

#include <random>
#include <primitiv/device.h>

namespace primitiv {

/**
 * Device class for the usual CPU.
 */
class CPUDevice : public Device {
  CPUDevice(const CPUDevice &) = delete;
  CPUDevice(CPUDevice &&) = delete;
  CPUDevice &operator=(const CPUDevice &) = delete;
  CPUDevice &operator=(CPUDevice &&) = delete;

public:
  /**
   * Creates a CPUDevice object.
   * @remarks The internal random number generator is initialized by
   *          `std::random_device`.
   */
  CPUDevice() : rng_(std::random_device()()) {}

  /**
   * Creates a CPUDevice object.
   * @param rng_seed The seed value of internal random number generator.
   */
  explicit CPUDevice(unsigned rng_seed) : rng_(rng_seed) {}

  ~CPUDevice() override = default;

  Device::DeviceType type() const override { return Device::DEVICE_TYPE_CPU; }

private:
  std::shared_ptr<void> new_handle(const Shape &shape) override;

  std::vector<float> tensor_to_vector_impl(const Tensor &x) override;

  void reset_tensor_impl(Tensor &x, float k) override;
  void reset_tensor_by_array_impl(Tensor &x, const float values[]) override;

  Tensor copy_tensor_impl(const Tensor &x) override;

  Tensor random_bernoulli_impl(const Shape &shape, float p) override;
  Tensor random_uniform_impl(
      const Shape &shape, float lower, float upper) override;
  Tensor random_normal_impl(const Shape &shape, float mean, float sd) override;
  Tensor random_log_normal_impl(
      const Shape &shape, float mean, float sd) override;

  Tensor pick_impl(
      const Tensor &x, unsigned dim,
      const std::vector<unsigned> &ids, Shape &&new_shape) override;
  Tensor slice_impl(
      const Tensor &x,
      unsigned dim, unsigned offset, Shape &&new_shape) override;
  Tensor concat_impl(
      const std::vector<const Tensor *> &xs,
      unsigned dim, Shape &&new_shape) override;

  Tensor negate_impl(const Tensor &x) override;

  Tensor add_impl(const Tensor &x, float k) override;
  Tensor add_impl(
      const Tensor &a, const Tensor &b, Shape &&new_shape) override;
  Tensor subtract_impl(const Tensor &x, float k) override;
  Tensor subtract_impl(float k, const Tensor &x) override;
  Tensor subtract_impl(
      const Tensor &a, const Tensor &b, Shape &&new_shape) override;
  Tensor multiply_impl(const Tensor &x, float k) override;
  Tensor multiply_impl(
      const Tensor &a, const Tensor &b, Shape &&new_shape) override;
  Tensor divide_impl(const Tensor &x, float k) override;
  Tensor divide_impl(float k, const Tensor &x) override;
  Tensor divide_impl(
      const Tensor &a, const Tensor &b, Shape &&new_shape) override;

  Tensor transpose_impl(const Tensor &x, Shape &&new_shape) override;
  Tensor dot_impl(const Tensor &a, const Tensor &b, Shape &&new_shape) override;

  Tensor sqrt_impl(const Tensor &x) override;
  Tensor exp_impl(const Tensor &x) override;
  Tensor tanh_impl(const Tensor &x) override;
  Tensor sigmoid_impl(const Tensor &x) override;
  Tensor step_impl(const Tensor &x) override;
  Tensor relu_impl(const Tensor &x) override;

  Tensor sum_impl(const Tensor &x, unsigned dim) override;
  Tensor logsumexp_impl(const Tensor &x, unsigned dim) override;
  Tensor broadcast_impl(
      const Tensor &x, unsigned dim, unsigned size, Shape &&new_shape) override;

  Tensor batch_sum_impl(const Tensor &x) override;

  void add_gradient_impl(Tensor &a, const Tensor &b) override;
  void add_gradient_offset_impl(
      Tensor &a, const Tensor &b, unsigned dim, unsigned offset) override;
  void add_gradient_sparse_impl(
      Tensor &a, const Tensor &b,
      unsigned dim, const std::vector<unsigned> &ids) override;

private:
  std::mt19937 rng_;
};

}  // namespace primitiv

#endif  // PRIMITIV_CPU_DEVICE_H_
