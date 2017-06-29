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

  void copy_tensor_impl(const Tensor &x, Tensor &y) override;

  void random_bernoulli_impl(float p, Tensor &y) override;
  void random_uniform_impl(float lower, float upper, Tensor &y) override;
  void random_normal_impl(float mean, float sd, Tensor &y) override;
  void random_log_normal_impl(float mean, float sd, Tensor &y) override;

  Tensor pick_fw_impl(const Tensor &x, unsigned dim, const std::vector<unsigned> &ids, Shape &&new_shape) override;
  Tensor slice_fw_impl(const Tensor &x, unsigned dim, unsigned offset, Shape &&new_shape) override;
  Tensor concat_fw_impl(const std::vector<const Tensor *> &xs, unsigned dim, Shape &&new_shape) override;

  Tensor negate_fw_impl(const Tensor &x) override;
  Tensor sqrt_fw_impl(const Tensor &x) override;
  Tensor exp_fw_impl(const Tensor &x) override;
  Tensor tanh_fw_impl(const Tensor &x) override;
  Tensor sigmoid_fw_impl(const Tensor &x) override;
  Tensor sin_fw_impl(const Tensor &x) override;
  Tensor cos_fw_impl(const Tensor &x) override;
  Tensor tan_fw_impl(const Tensor &x) override;

  Tensor add_const_fw_impl(const Tensor &x, float k) override;
  Tensor subtract_const_r_fw_impl(const Tensor &x, float k) override;
  Tensor subtract_const_l_fw_impl(const Tensor &x, float k) override;
  Tensor multiply_const_fw_impl(const Tensor &x, float k) override;
  Tensor divide_const_r_fw_impl(const Tensor &x, float k) override;
  Tensor divide_const_l_fw_impl(const Tensor &x, float k) override;
  Tensor pstep_fw_impl(const Tensor &x, float k) override;
  Tensor prelu_fw_impl(const Tensor &x, float k) override;

  Tensor add_scalar_fw_impl(const Tensor &x, const Tensor &k, Shape &&new_shape) override;
  Tensor subtract_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Shape &&new_shape) override;
  Tensor subtract_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Shape &&new_shape) override;
  Tensor multiply_scalar_fw_impl(const Tensor &x, const Tensor &k, Shape &&new_shape) override;
  Tensor divide_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Shape &&new_shape) override;
  Tensor divide_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Shape &&new_shape) override;

  Tensor add_fw_impl(const Tensor &a, const Tensor &b, Shape &&new_shape) override;
  Tensor subtract_fw_impl(const Tensor &a, const Tensor &b, Shape &&new_shape) override;
  Tensor multiply_fw_impl(const Tensor &a, const Tensor &b, Shape &&new_shape) override;
  Tensor divide_fw_impl(const Tensor &a, const Tensor &b, Shape &&new_shape) override;

  Tensor transpose_fw_impl(const Tensor &x, Shape &&new_shape) override;
  Tensor matmul_fw_impl(const Tensor &a, const Tensor &b, Shape &&new_shape) override;
  void matmul_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &gy,
      Tensor &ga, Tensor &gb) override;

  Tensor sum_fw_impl(const Tensor &x, unsigned dim) override;
  Tensor logsumexp_fw_impl(const Tensor &x, unsigned dim) override;
  Tensor broadcast_fw_impl(const Tensor &x, unsigned dim, unsigned size, Shape &&new_shape) override;
  Tensor batch_sum_fw_impl(const Tensor &x) override;

  void add_gradient_impl(Tensor &a, const Tensor &b) override;
  void add_gradient_offset_impl(Tensor &a, const Tensor &b, unsigned dim, unsigned offset) override;
  void add_gradient_sparse_impl(Tensor &a, const Tensor &b, unsigned dim, const std::vector<unsigned> &ids) override;

private:
  std::mt19937 rng_;
};

}  // namespace primitiv

#endif  // PRIMITIV_CPU_DEVICE_H_
