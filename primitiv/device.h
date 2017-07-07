#ifndef PRIMITIV_DEVICE_H_
#define PRIMITIV_DEVICE_H_

#include <memory>
#include <primitiv/shape.h>
#include <primitiv/tensor.h>

namespace primitiv {

/**
 * Interface of the Tensor provider.
 */
class Device {
  friend Tensor;

  Device(const Device &) = delete;
  Device(Device &&) = delete;
  Device &operator=(const Device &) = delete;
  Device &operator=(Device &&) = delete;

public:
  /**
   * Device type.
   */
  enum DeviceType {
    DEVICE_TYPE_CPU = 0x0,
    DEVICE_TYPE_CUDA = 0x10000,
  };

  Device() = default;
  virtual ~Device() = default;

  /**
   * Retrieves the type of the device.
   * @return A DeviceType value.
   */
  virtual DeviceType type() const = 0;

  /**
   * Provides a new Tensor object on the device.
   * @param shape Shape of the tensor.
   * @return A new Tensor object.
   */
  Tensor new_tensor(const Shape &shape);

  /**
   * Provides a new Tensor object with same-value elements.
   * @param shape Shape of the tensor.
   * @param k Constant to initialize elements.
   * @return A new Tensor object.
   */
  Tensor new_tensor(const Shape &shape, float k);

  /**
   * Provides a new Tensor object with specific values.
   * @param shape Shape of the tensor.
   * @param values Pointer to array of internal values.
   * @return A new Tensor object.
   */
  Tensor new_tensor_by_array(const Shape &shape, const float values[]);

  /**
   * Provides a new Tensor object with specific values.
   * @param shape Shape of the tensor.
   * @param values List of internal values.
   * @return A new Tensor object.
   */
  Tensor new_tensor_by_vector(
      const Shape &shape, const std::vector<float> &values);

  /**
   * Copies the tensor to this device with allocating a new memory.
   * @param x A tensor to be copied.
   * @return Copied tensor.
   * @remarks The value of `x` is always duplicated, and the internal memory of
   *          the resulting tensor becomes always different from `x` even if
   *          `x.device()` is same as `this`.
   */
  Tensor copy_tensor(const Tensor &x);

  // Random value generators.
  Tensor random_bernoulli(const Shape &shape, float p);
  Tensor random_uniform(const Shape &shape, float lower, float upper);
  Tensor random_normal(const Shape &shape, float mean, float sd);
  Tensor random_log_normal(const Shape &shape, float mean, float sd);

  // Tensor manipulations.
  Tensor pick_fw(const Tensor &x, unsigned dim, const std::vector<unsigned> &ids);
  Tensor slice_fw(const Tensor &x, unsigned dim, unsigned lower, unsigned upper);
  Tensor concat_fw(const std::vector<const Tensor *> &xs, unsigned dim);

  void pick_bw(const Tensor &gy, unsigned dim, const std::vector<unsigned> &ids, Tensor &gx);
  void slice_bw(const Tensor &gy, unsigned dim, unsigned offset, Tensor &gx);

  // Unary operations.
  Tensor negate_fw(const Tensor &x);
  Tensor sqrt_fw(const Tensor &x);
  Tensor exp_fw(const Tensor &x);
  Tensor tanh_fw(const Tensor &x);
  Tensor sigmoid_fw(const Tensor &x);
  Tensor sin_fw(const Tensor &x);
  Tensor cos_fw(const Tensor &x);
  Tensor tan_fw(const Tensor &x);
  Tensor transpose_fw(const Tensor &x);

  void negate_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void sqrt_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void exp_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void tanh_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void sigmoid_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void sin_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void cos_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void tan_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void transpose_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);

  // Tensor-constant operations.
  Tensor add_const_fw(const Tensor &x, float k);
  Tensor subtract_const_r_fw(const Tensor &x, float k);
  Tensor subtract_const_l_fw(const Tensor &x, float k);
  Tensor multiply_const_fw(const Tensor &x, float k);
  Tensor divide_const_r_fw(const Tensor &x, float k);
  Tensor divide_const_l_fw(const Tensor &x, float k);
  Tensor prelu_fw(const Tensor &x, float k);
  Tensor elu_fw(const Tensor &x, float k);

  void add_const_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void subtract_const_r_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void subtract_const_l_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void multiply_const_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void divide_const_r_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void divide_const_l_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void prelu_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void elu_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);

  // Tensor-scalar operations.
  Tensor add_scalar_fw(const Tensor &x, const Tensor &k);
  Tensor subtract_scalar_r_fw(const Tensor &x, const Tensor &k);
  Tensor subtract_scalar_l_fw(const Tensor &x, const Tensor &k);
  Tensor multiply_scalar_fw(const Tensor &x, const Tensor &k);
  Tensor divide_scalar_r_fw(const Tensor &x, const Tensor &k);
  Tensor divide_scalar_l_fw(const Tensor &x, const Tensor &k);

  // Binary operations.
  Tensor add_fw(const Tensor &a, const Tensor &b);
  Tensor subtract_fw(const Tensor &a, const Tensor &b);
  Tensor multiply_fw(const Tensor &a, const Tensor &b);
  Tensor divide_fw(const Tensor &a, const Tensor &b);
  Tensor matmul_fw(const Tensor &a, const Tensor &b);

  void add_bw(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb);
  void subtract_bw(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb);
  void multiply_bw(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb);
  void divide_bw(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb);
  void matmul_bw(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb);

  // Dimension operations.
  Tensor sum_fw(const Tensor &x, unsigned dim);
  Tensor logsumexp_fw(const Tensor &x, unsigned dim);
  Tensor broadcast_fw(const Tensor &x, unsigned dim, unsigned size);
  Tensor batch_sum_fw(const Tensor &x);

  /**
   * Directly adds the second tensor to the first tensor.
   * @param gy A source tensor.
   * @param gx A tensor to be udpated.
   * @remarks This method keeps the shape of `gx`, and the behavior is
   *          conditioned according to the batch size of `gx` and `gy`:
   *              gx.shape == gy.shape: gx += gy
   *              gx.shape == 1:        gx += batch_sum(gy)
   *              gy.shape == 1:        gx += batch_broadcast(gy)
   *              otherwise: error.
   */
  void inplace_add(const Tensor &gy, Tensor &gx);

private:
  /**
   * Retrieves internal values of the tensor as a vector.
   * @param x A tensor.
   * @return A list of the internal values.
   * @remarks Each resulting values are ordered by the column-major order, and
   *          the batch size is assumed as the last dimension of the tensor.
   */
  std::vector<float> tensor_to_vector(const Tensor &x);

protected:
  /**
   * Reset internal values of the tensor using a constant.
   * @param k A value used to initialize each element.
   * @param x A tensor to be updated.
   */
  void reset_tensor(float k, Tensor &x);

  /**
   * Reset internal values of the tensor using specific values.
   * @param values Array of each element.
   * @param x A tensor to be updated.
   * @remarks `values.size()` should be same as `x.shape().size()`. Each element
   *          is ordered by the column-major order, and the batch size is
   *          assumed as the last dimension of the tensor.
   */
  void reset_tensor_by_array(const float values[], Tensor &x);

  /**
   * Reset internal values of the tensor using specific values.
   * @param values List of each element.
   * @param x A tensor to be updated.
   * @remarks `values.size()` should be same as `x.shape().size()`. Each element
   *          is ordered by the column-major order, and the batch size is
   *          assumed as the last dimension of the tensor.
   */
  void reset_tensor_by_vector(const std::vector<float> &values, Tensor &x);

private:
  // device-specific implementations.

  virtual std::shared_ptr<void> new_handle(const Shape &shape) = 0;

  virtual std::vector<float> tensor_to_vector_impl(const Tensor &x) = 0;

  virtual void reset_tensor_impl(float k, Tensor &x) = 0;
  virtual void reset_tensor_by_array_impl(const float values[], Tensor &x) = 0;

  virtual void copy_tensor_impl(const Tensor &x, Tensor &y) = 0;

  virtual void random_bernoulli_impl(float p, Tensor &y) = 0;
  virtual void random_uniform_impl(float lower, float upper, Tensor &y) = 0;
  virtual void random_normal_impl(float mean, float sd, Tensor &y) = 0;
  virtual void random_log_normal_impl(float mean, float sd, Tensor &y) = 0;

  virtual void pick_fw_impl(const Tensor &x, unsigned dim, const std::vector<unsigned> &ids, Tensor &y) = 0;
  virtual void slice_fw_impl(const Tensor &x, unsigned dim, unsigned offset, Tensor &y) = 0;
  virtual void concat_fw_impl(const std::vector<const Tensor *> &xs, unsigned dim, Tensor &y) = 0;

  virtual void pick_bw_impl(const Tensor &gy, unsigned dim, const std::vector<unsigned> &ids, Tensor &gx) = 0;
  virtual void slice_bw_impl(const Tensor &gy, unsigned dim, unsigned offset, Tensor &gx) = 0;

  virtual void negate_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void sqrt_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void exp_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void tanh_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void sigmoid_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void sin_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void cos_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void tan_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void transpose_fw_impl(const Tensor &x, Tensor &y) = 0;

  virtual void negate_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void sqrt_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void exp_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void tanh_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void sigmoid_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void sin_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void cos_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void tan_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void transpose_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;

  virtual void add_const_fw_impl(const Tensor &x, float k, Tensor &y) = 0;
  virtual void subtract_const_r_fw_impl(const Tensor &x, float k, Tensor &y) = 0;
  virtual void subtract_const_l_fw_impl(const Tensor &x, float k, Tensor &y) = 0;
  virtual void multiply_const_fw_impl(const Tensor &x, float k, Tensor &y) = 0;
  virtual void divide_const_r_fw_impl(const Tensor &x, float k, Tensor &y) = 0;
  virtual void divide_const_l_fw_impl(const Tensor &x, float k, Tensor &y)  = 0;
  virtual void prelu_fw_impl(const Tensor &x, float k, Tensor &y) = 0;
  virtual void elu_fw_impl(const Tensor &x, float k, Tensor &y) = 0;

  virtual void add_const_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void subtract_const_r_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void subtract_const_l_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void multiply_const_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void divide_const_r_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void divide_const_l_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void prelu_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void elu_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;

  virtual void add_scalar_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) = 0;
  virtual void subtract_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) = 0;
  virtual void subtract_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) = 0;
  virtual void multiply_scalar_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) = 0;
  virtual void divide_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) = 0;
  virtual void divide_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) = 0;

  virtual void add_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) = 0;
  virtual void subtract_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) = 0;
  virtual void multiply_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) = 0;
  virtual void divide_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) = 0;
  virtual void matmul_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) = 0;

  virtual void add_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) = 0;
  virtual void subtract_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) = 0;
  virtual void multiply_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) = 0;
  virtual void divide_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) = 0;
  virtual void matmul_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) = 0;

  virtual void sum_fw_impl(const Tensor &x, unsigned dim, Tensor &y) = 0;
  virtual void logsumexp_fw_impl(const Tensor &x, unsigned dim, Tensor &y) = 0;
  virtual void broadcast_fw_impl(const Tensor &x, unsigned dim, unsigned size, Tensor &y) = 0;
  virtual void batch_sum_fw_impl(const Tensor &x, Tensor &y) = 0;

  virtual void inplace_add_impl(const Tensor &gy, Tensor &gx) = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_DEVICE_H_
