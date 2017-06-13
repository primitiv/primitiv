#ifndef PRIMITIV_CPU_DEVICE_H_
#define PRIMITIV_CPU_DEVICE_H_

#include <map>
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
  CPUDevice();

  /**
   * Creates a CPUDevice object.
   * @param rng_seed The seed value of internal random number generator.
   */
  explicit CPUDevice(unsigned rng_seed);

  ~CPUDevice() override;

private:
  void *new_handle(const Shape &shape) override;
  void delete_tensor_impl(Tensor &x) override;

  std::vector<float> tensor_to_vector_impl(const Tensor &x) override;

  void reset_tensor_impl(Tensor &x, float k) override;
  void reset_tensor_impl(Tensor &x, const std::vector<float> &values) override;

  Tensor random_bernoulli_impl(const Shape &shape, float p) override;
  Tensor random_uniform_impl(
      const Shape &shape, float lower, float upper) override;
  Tensor random_normal_impl(const Shape &shape, float mean, float sd) override;

  Tensor slice_impl(
      const Tensor &x,
      unsigned dim, unsigned offset, const Shape &new_shape) override;
  Tensor concat_impl(
      const std::vector<const Tensor *> &xs,
      unsigned dim, const Shape &new_shape) override;

  Tensor duplicate_impl(const Tensor &x) override;
  Tensor negate_impl(const Tensor &x) override;

  Tensor add_impl(const Tensor &x, float k) override;
  Tensor add_impl(const Tensor &a, const Tensor &b) override;
  Tensor subtract_impl(const Tensor &x, float k) override;
  Tensor subtract_impl(float k, const Tensor &x) override;
  Tensor subtract_impl(const Tensor &a, const Tensor &b) override;
  Tensor multiply_impl(const Tensor &x, float k) override;
  Tensor multiply_impl(const Tensor &a, const Tensor &b) override;
  Tensor divide_impl(const Tensor &x, float k) override;
  Tensor divide_impl(float k, const Tensor &x) override;
  Tensor divide_impl(const Tensor &a, const Tensor &b) override;

  Tensor transpose_impl(const Tensor &x) override;
  Tensor dot_impl(const Tensor &a, const Tensor &b) override;

  Tensor exp_impl(const Tensor &x) override;
  Tensor tanh_impl(const Tensor &x) override;
  Tensor sigmoid_impl(const Tensor &x) override;
  Tensor step_impl(const Tensor &x) override;
  Tensor relu_impl(const Tensor &x) override;

  Tensor sum_impl(const Tensor &x, unsigned dim) override;
  Tensor broadcast_impl(const Tensor &x, unsigned dim, unsigned size) override;

  Tensor batch_sum_impl(const Tensor &x) override;

  void add_gradient_impl(Tensor &a, const Tensor &b) override;
  void add_gradient_offset_impl(
      Tensor &a, const Tensor &b, unsigned dim, unsigned offset) override;

private:
  std::map<void *, unsigned> blocks_;
  std::mt19937 rng_;
};

}  // namespace primitiv

#endif  // PRIMITIV_CPU_DEVICE_H_
