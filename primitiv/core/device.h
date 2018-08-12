#ifndef PRIMITIV_CORE_DEVICE_H_
#define PRIMITIV_CORE_DEVICE_H_

#include <cstdint>
#include <memory>

#include <primitiv/core/mixins/default_settable.h>
#include <primitiv/core/mixins/nonmovable.h>
#include <primitiv/core/shape.h>
#include <primitiv/core/tensor.h>

namespace primitiv {

/**
 * Device type.
 */
enum class DeviceType : std::uint32_t {
  GROUP_FILTER = 0xffff0000,

  GROUP_CPU = 0x00000000,

  // NOTE(odashi):
  // DeviceType::CPU is deprecated and will be deleted in the next release.
  CPU = 0x00000000,

  NAIVE = 0x00000000,
  EIGEN = 0x00000001,

  GROUP_CUDA = 0x00010000,
  CUDA = 0x00010000,
  CUDA16 = 0x00010001,

  GROUP_OPENCL = 0x00020000,
  OPENCL = 0x00020000,
};

/**
 * Interface of the Tensor provider.
 */
class Device
    : public mixins::DefaultSettable<Device>
    , mixins::Nonmovable<Device> {
  friend Tensor;

public:
  Device() = default;
  virtual ~Device() = default;

  /**
   * Prints device description to stderr.
   */
  virtual void dump_description() const = 0;

  /**
   * Retrieves the type of the device.
   * @return A DeviceType value.
   */
  virtual DeviceType type() const = 0;

private:
  /**
   * Provides a new Tensor object on the device.
   * @param shape Shape of the tensor.
   * @return A new Tensor object.
   */
  Tensor new_raw_tensor(const Shape &shape);

public:
  /**
   * Provides a new Tensor object with same-value elements.
   * @param shape Shape of the tensor.
   * @param k Constant to initialize elements.
   * @return A new Tensor object.
   */
  Tensor new_tensor_by_constant(const Shape &shape, float k);

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

  // Provides an identity matrix.
  Tensor identity(std::uint32_t size);

  // Random value generators.
  Tensor random_bernoulli(const Shape &shape, float p);
  Tensor random_uniform(const Shape &shape, float lower, float upper);  // (lower, upper]
  Tensor random_normal(const Shape &shape, float mean, float sd);
  Tensor random_log_normal(const Shape &shape, float mean, float sd);

  // Tensor manipulations.
  Tensor pick_fw(const Tensor &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim);
  Tensor slice_fw(const Tensor &x, std::uint32_t dim, std::uint32_t lower, std::uint32_t upper);
  Tensor concat_fw(const std::vector<const Tensor *> &xs, std::uint32_t dim);

  void pick_bw(const Tensor &gy, const std::vector<std::uint32_t> &ids, std::uint32_t dim, Tensor &gx);
  void slice_bw(const Tensor &gy, std::uint32_t dim, std::uint32_t offset, Tensor &gx);

  // Unary operations.
  Tensor negate_fw(const Tensor &x);
  Tensor abs_fw(const Tensor &x);
  Tensor sqrt_fw(const Tensor &x);
  Tensor exp_fw(const Tensor &x);
  Tensor log_fw(const Tensor &x);
  Tensor tanh_fw(const Tensor &x);
  Tensor sigmoid_fw(const Tensor &x);
  Tensor softplus_fw(const Tensor &x);
  Tensor sin_fw(const Tensor &x);
  Tensor cos_fw(const Tensor &x);
  Tensor tan_fw(const Tensor &x);
  Tensor transpose_fw(const Tensor &x);
  Tensor permute_dims_fw(const Tensor &x, const std::vector<std::uint32_t> &perm);

  Tensor flip_fw(const Tensor &x, std::uint32_t dim);

  void abs_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void sqrt_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void exp_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void log_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void tanh_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void sigmoid_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void softplus_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void sin_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void cos_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void tan_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void transpose_bw(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx);
  void permute_dims_bw(const Tensor &x, const Tensor &y, const Tensor &gy, const std::vector<std::uint32_t> &perm, Tensor &gx);

  void flip_bw(const Tensor &gy, std::uint32_t dim, Tensor &gx);

  // Tensor-constant operations.
  Tensor add_const_fw(const Tensor &x, float k);
  Tensor subtract_const_r_fw(const Tensor &x, float k);
  Tensor subtract_const_l_fw(const Tensor &x, float k);
  Tensor multiply_const_fw(const Tensor &x, float k);
  Tensor divide_const_r_fw(const Tensor &x, float k);
  Tensor divide_const_l_fw(const Tensor &x, float k);
  Tensor pow_const_r_fw(const Tensor &x, float k);
  Tensor pow_const_l_fw(const Tensor &x, float k);
  Tensor prelu_fw(const Tensor &x, float k);
  Tensor elu_fw(const Tensor &x, float k);

  Tensor pown_fw(const Tensor &x, std::int32_t k);

  void add_const_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void subtract_const_r_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void subtract_const_l_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void multiply_const_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void divide_const_r_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void divide_const_l_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void pow_const_r_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void pow_const_l_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void prelu_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);
  void elu_bw(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx);

  void pown_bw(const Tensor &x, const Tensor &y, const Tensor &gy, std::int32_t k, Tensor &gx);

  // Tensor-scalar operations.
  Tensor add_scalar_fw(const Tensor &x, const Tensor &k);
  Tensor subtract_scalar_r_fw(const Tensor &x, const Tensor &k);
  Tensor subtract_scalar_l_fw(const Tensor &x, const Tensor &k);
  Tensor multiply_scalar_fw(const Tensor &x, const Tensor &k);
  Tensor divide_scalar_r_fw(const Tensor &x, const Tensor &k);
  Tensor divide_scalar_l_fw(const Tensor &x, const Tensor &k);
  Tensor pow_scalar_r_fw(const Tensor &x, const Tensor &k);
  Tensor pow_scalar_l_fw(const Tensor &x, const Tensor &k);

  // Binary operations.
  Tensor add_fw(const Tensor &a, const Tensor &b);
  Tensor subtract_fw(const Tensor &a, const Tensor &b);
  Tensor multiply_fw(const Tensor &a, const Tensor &b);
  Tensor divide_fw(const Tensor &a, const Tensor &b);
  Tensor pow_fw(const Tensor &a, const Tensor &b);
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
  void pow_bw(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb);
  void matmul_bw(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb);

  // Dimension operations.
  Tensor max_fw(const Tensor &x, std::uint32_t dim);
  Tensor min_fw(const Tensor &x, std::uint32_t dim);
  void max_bw(const Tensor &x, const Tensor &y, const Tensor &gy, std::uint32_t dim, Tensor &gx);
  void min_bw(const Tensor &x, const Tensor &y, const Tensor &gy, std::uint32_t dim, Tensor &gx);

  Tensor sum_fw(const Tensor &x, std::uint32_t dim);
  Tensor logsumexp_fw(const Tensor &x, std::uint32_t dim);
  Tensor broadcast_fw(const Tensor &x, std::uint32_t dim, std::uint32_t size);

  // Minibatch operations.
  Tensor batch_pick_fw(const Tensor &x, const std::vector<std::uint32_t> &ids);
  Tensor batch_slice_fw(const Tensor &x, std::uint32_t lower, std::uint32_t upper);
  Tensor batch_concat_fw(const std::vector<const Tensor *> &xs);
  Tensor batch_sum_fw(const Tensor &x);

  void batch_pick_bw(const Tensor &gy, const std::vector<std::uint32_t> &ids, Tensor &gx);
  void batch_slice_bw(const Tensor &gy, std::uint32_t offset, Tensor &gx);

  // Convolution.
  Tensor conv2d_fw(
      const Tensor &x, const Tensor &w,
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1,
      std::uint32_t dilation0, std::uint32_t dilation1);

  void conv2d_bw(
      const Tensor &x, const Tensor &w, const Tensor &y, const Tensor &gy,
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1,
      std::uint32_t dilation0, std::uint32_t dilation1,
      Tensor &gx, Tensor &gw);

  // Pooling.
  Tensor max_pool2d_fw(
      const Tensor &x,
      std::uint32_t window0, std::uint32_t window1,
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1);

  void max_pool2d_bw(
      const Tensor &x, const Tensor &y, const Tensor &gy,
      std::uint32_t window0, std::uint32_t window1,
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1,
      Tensor &gx);

  /**
   * Directly multiplies all elements by a constant.
   * @param k A constant to multiply.
   * @param x A tensor to be updated.
   */
  void inplace_multiply_const(float k, Tensor &x);

  /**
   * Directly adds the first tensor to the second tensor.
   * @param x A tensor to add.
   * @param y A tensor to be udpated.
   * @remarks This method keeps the shape of `y`, and the behavior is
   *          conditioned according to the batch size of `y` and `x`:
   *              y.shape == x.shape: y += x
   *              y.shape == 1:       y += batch_sum(x)
   *              x.shape == 1:       y += batch_broadcast(x)
   *              otherwise: error.
   */
  void inplace_add(const Tensor &x, Tensor &y);

  /**
   * Directly subtracts the first tensor from the second tensor.
   * @param x A tensor to subtract.
   * @param y A tensor to be updated.
   * @remarks The batch broadcasting behavior of this method is same as that of
   *          `inplace_add`.
   */
  void inplace_subtract(const Tensor &x, Tensor &y);

private:
  /**
   * Retrieves internal values of the tensor as a vector.
   * @param x A tensor.
   * @return A list of the internal values.
   * @remarks Each resulting values are ordered by the column-major order, and
   *          the batch size is assumed as the last dimension of the tensor.
   */
  std::vector<float> tensor_to_vector(const Tensor &x);

  /**
   * Retrieves argmax indices along an axis.
   * @param x A tensor.
   * @param dim A specified axis.
   * @return A list of integers that indicates positions of the maximum values.
   */
  std::vector<std::uint32_t> argmax(const Tensor &x, std::uint32_t dim);

  /**
   * Retrieves argmin indices along an axis.
   * @param x A tensor.
   * @param dim A specified axis.
   * @return A list of integers that indicates positions of the minimum values.
   */
  std::vector<std::uint32_t> argmin(const Tensor &x, std::uint32_t dim);

protected:
  /**
   * Obtains an inner handle from a Tensor.
   * @param x Target Tensor object.
   * @return Inner handle of `x`.
   */
  static const void *get_handle(const Tensor &x) {
    return x.handle();
  }

  /**
   * Obtains a mutable inner handle from a Tensor.
   * @param x Target Tensor object.
   * @return Mutable inner handle of `x`.
   */
  static void *get_mutable_handle(Tensor &x) {
    return x.mutable_handle();
  }

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
  virtual std::vector<std::uint32_t> argmax_impl(const Tensor &x, std::uint32_t dim) = 0;
  virtual std::vector<std::uint32_t> argmin_impl(const Tensor &x, std::uint32_t dim) = 0;

  virtual void reset_tensor_impl(float k, Tensor &x) = 0;
  virtual void reset_tensor_by_array_impl(const float values[], Tensor &x) = 0;

  virtual void copy_tensor_impl(const Tensor &x, Tensor &y) = 0;

  virtual void identity_impl(Tensor &y) = 0;

  virtual void random_bernoulli_impl(float p, Tensor &y) = 0;
  virtual void random_uniform_impl(float lower, float upper, Tensor &y) = 0;
  virtual void random_normal_impl(float mean, float sd, Tensor &y) = 0;
  virtual void random_log_normal_impl(float mean, float sd, Tensor &y) = 0;

  virtual void pick_fw_impl(const Tensor &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim, Tensor &y) = 0;
  virtual void slice_fw_impl(const Tensor &x, std::uint32_t dim, std::uint32_t offset, Tensor &y) = 0;
  virtual void concat_fw_impl(const std::vector<const Tensor *> &xs, std::uint32_t dim, Tensor &y) = 0;

  virtual void pick_bw_impl(const Tensor &gy, const std::vector<std::uint32_t> &ids, std::uint32_t dim, Tensor &gx) = 0;
  virtual void slice_bw_impl(const Tensor &gy, std::uint32_t dim, std::uint32_t offset, Tensor &gx) = 0;

  virtual void negate_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void abs_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void sqrt_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void exp_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void log_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void tanh_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void sigmoid_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void softplus_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void sin_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void cos_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void tan_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void transpose_fw_impl(const Tensor &x, Tensor &y) = 0;
  virtual void permute_dims_fw_impl(const Tensor &x, const std::vector<std::uint32_t> &perm, Tensor &y) = 0;

  virtual void flip_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) = 0;

  virtual void abs_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void sqrt_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void exp_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void log_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void tanh_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void sigmoid_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void softplus_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void sin_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void cos_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void tan_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void transpose_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) = 0;
  virtual void permute_dims_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, const std::vector<std::uint32_t> &perm, Tensor &gx) = 0;

  virtual void flip_bw_impl(const Tensor &gy, std::uint32_t dim, Tensor &gx) = 0;

  virtual void add_const_fw_impl(const Tensor &x, float k, Tensor &y) = 0;
  virtual void subtract_const_r_fw_impl(const Tensor &x, float k, Tensor &y) = 0;
  virtual void subtract_const_l_fw_impl(const Tensor &x, float k, Tensor &y) = 0;
  virtual void multiply_const_fw_impl(const Tensor &x, float k, Tensor &y) = 0;
  virtual void divide_const_r_fw_impl(const Tensor &x, float k, Tensor &y) = 0;
  virtual void divide_const_l_fw_impl(const Tensor &x, float k, Tensor &y)  = 0;
  virtual void pow_const_r_fw_impl(const Tensor &x, float k, Tensor &y) = 0;
  virtual void pow_const_l_fw_impl(const Tensor &x, float k, Tensor &y)  = 0;
  virtual void prelu_fw_impl(const Tensor &x, float k, Tensor &y) = 0;
  virtual void elu_fw_impl(const Tensor &x, float k, Tensor &y) = 0;

  virtual void pown_fw_impl(const Tensor &x, std::int32_t k, Tensor &y) = 0;

  virtual void add_const_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void subtract_const_r_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void subtract_const_l_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void multiply_const_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void divide_const_r_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void divide_const_l_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void pow_const_r_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void pow_const_l_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void prelu_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;
  virtual void elu_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) = 0;

  virtual void pown_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, std::int32_t k, Tensor &gx) = 0;

  virtual void add_scalar_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) = 0;
  virtual void subtract_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) = 0;
  virtual void subtract_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) = 0;
  virtual void multiply_scalar_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) = 0;
  virtual void divide_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) = 0;
  virtual void divide_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) = 0;
  virtual void pow_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) = 0;
  virtual void pow_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) = 0;

  virtual void add_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) = 0;
  virtual void subtract_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) = 0;
  virtual void multiply_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) = 0;
  virtual void divide_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) = 0;
  virtual void pow_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) = 0;
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
  virtual void pow_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) = 0;
  virtual void matmul_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) = 0;

  virtual void max_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) = 0;
  virtual void min_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) = 0;
  virtual void max_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, std::uint32_t dim, Tensor &gx) = 0;
  virtual void min_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, std::uint32_t dim, Tensor &gx) = 0;

  virtual void sum_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) = 0;
  virtual void logsumexp_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) = 0;
  virtual void broadcast_fw_impl(const Tensor &x, std::uint32_t dim, std::uint32_t size, Tensor &y) = 0;

  virtual void batch_pick_fw_impl(const Tensor &x, const std::vector<std::uint32_t> &ids, Tensor &y) = 0;
  virtual void batch_slice_fw_impl(const Tensor &x, std::uint32_t offset, Tensor &y) = 0;
  virtual void batch_concat_fw_impl(
      const std::vector<const Tensor *> &xs, Tensor &y) = 0;
  virtual void batch_sum_fw_impl(const Tensor &x, Tensor &y) = 0;

  virtual void batch_pick_bw_impl(const Tensor &gy, const std::vector<std::uint32_t> &ids, Tensor &gx) = 0;
  virtual void batch_slice_bw_impl(const Tensor &gy, std::uint32_t offset, Tensor &gx) = 0;

  virtual void conv2d_fw_impl(
      const Tensor &x, const Tensor &w,
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1,
      std::uint32_t dilation0, std::uint32_t dilation1,
      Tensor &y) = 0;

  virtual void conv2d_bw_impl(
      const Tensor &x, const Tensor &w, const Tensor &y, const Tensor &gy,
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1,
      std::uint32_t dilation0, std::uint32_t dilation1,
      Tensor &gx, Tensor &gw) = 0;

  virtual void max_pool2d_fw_impl(
      const Tensor &x,
      std::uint32_t window0, std::uint32_t window1,
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1,
      Tensor &y) = 0;

  virtual void max_pool2d_bw_impl(
      const Tensor &x, const Tensor &y, const Tensor &gy,
      std::uint32_t window0, std::uint32_t window1,
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1,
      Tensor &gx) = 0;

  virtual void inplace_multiply_const_impl(float k, Tensor &x) = 0;

  virtual void inplace_add_impl(const Tensor &x, Tensor &y) = 0;
  virtual void inplace_subtract_impl(const Tensor &x, Tensor &y) = 0;
};

}  // namespace primitiv

#endif  // PRIMITIV_CORE_DEVICE_H_
