#ifndef PRIMITIV_DEVICES_NAIVE_DEVICE_H_
#define PRIMITIV_DEVICES_NAIVE_DEVICE_H_

#include <primitiv/core/device.h>
#include <primitiv/core/random.h>

namespace primitiv {
namespace devices {

/**
 * Device class for the naive function implementations on CPU.
 */
class Naive : public Device {
public:
  /**
   * Creates a Naive object.
   */
  Naive() = default;

  /**
   * Creates a Naive object.
   * @param seed The seed value of internal random number generator.
   */
  explicit Naive(std::uint32_t seed) : randomizer_(seed) {}

  ~Naive() override = default;

  void dump_description() const override;
  DeviceType type() const override { return DeviceType::NAIVE; }

private:
  std::shared_ptr<void> new_handle(const Shape &shape) override;

  std::vector<float> tensor_to_vector_impl(const Tensor &x) override;
  std::vector<std::uint32_t> argmax_impl(const Tensor &x, std::uint32_t dim) override;
  std::vector<std::uint32_t> argmin_impl(const Tensor &x, std::uint32_t dim) override;

  void reset_tensor_impl(float k, Tensor &x) override;
  void reset_tensor_by_array_impl(const float values[], Tensor &x) override;

  void copy_tensor_impl(const Tensor &x, Tensor &y) override;

  void identity_impl(Tensor &y) override;

  void random_bernoulli_impl(float p, Tensor &y) override;
  void random_uniform_impl(float lower, float upper, Tensor &y) override;
  void random_normal_impl(float mean, float sd, Tensor &y) override;
  void random_log_normal_impl(float mean, float sd, Tensor &y) override;

  void pick_fw_impl(const Tensor &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim, Tensor &y) override;
  void slice_fw_impl(const Tensor &x, std::uint32_t dim, std::uint32_t offset, Tensor &y) override;
  void concat_fw_impl(const std::vector<const Tensor *> &xs, std::uint32_t dim, Tensor &y) override;

  void pick_bw_impl(const Tensor &gy, const std::vector<std::uint32_t> &ids, std::uint32_t dim, Tensor &gx) override;
  void slice_bw_impl(const Tensor &gy, std::uint32_t dim, std::uint32_t offset, Tensor &gx) override;

  void negate_fw_impl(const Tensor &x, Tensor &y) override;
  void abs_fw_impl(const Tensor &x, Tensor &y) override;
  void sqrt_fw_impl(const Tensor &x, Tensor &y) override;
  void exp_fw_impl(const Tensor &x, Tensor &y) override;
  void log_fw_impl(const Tensor &x, Tensor &y) override;
  void tanh_fw_impl(const Tensor &x, Tensor &y) override;
  void sigmoid_fw_impl(const Tensor &x, Tensor &y) override;
  void softplus_fw_impl(const Tensor &x, Tensor &y) override;
  void sin_fw_impl(const Tensor &x, Tensor &y) override;
  void cos_fw_impl(const Tensor &x, Tensor &y) override;
  void tan_fw_impl(const Tensor &x, Tensor &y) override;
  void transpose_fw_impl(const Tensor &x, Tensor &y) override;
  void permute_dims_fw_impl(const Tensor &x, const std::vector<std::uint32_t> &perm, Tensor &y) override;

  void flip_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) override;

  void abs_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override;
  void sqrt_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override;
  void exp_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override;
  void log_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override;
  void tanh_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override;
  void sigmoid_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override;
  void softplus_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override;
  void sin_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override;
  void cos_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override;
  void tan_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override;
  void transpose_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override;
  void permute_dims_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, const std::vector<std::uint32_t> &perm, Tensor &gx) override;

  void flip_bw_impl(const Tensor &gy, std::uint32_t dim, Tensor &gx) override;

  void add_const_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void subtract_const_r_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void subtract_const_l_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void multiply_const_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void divide_const_r_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void divide_const_l_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void pow_const_r_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void pow_const_l_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void prelu_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void elu_fw_impl(const Tensor &x, float k, Tensor &y) override;

  void pown_fw_impl(const Tensor &x, std::int32_t k, Tensor &y) override;

  void add_const_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void subtract_const_r_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void subtract_const_l_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void multiply_const_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void divide_const_r_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void divide_const_l_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void pow_const_r_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void pow_const_l_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void prelu_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void elu_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;

  void pown_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, std::int32_t k, Tensor &gx) override;

  void add_scalar_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override;
  void subtract_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override;
  void subtract_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override;
  void multiply_scalar_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override;
  void divide_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override;
  void divide_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override;
  void pow_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override;
  void pow_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override;

  void add_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override;
  void subtract_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override;
  void multiply_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override;
  void divide_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override;
  void pow_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override;
  void matmul_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override;

  void add_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) override;
  void subtract_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) override;
  void multiply_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) override;
  void divide_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) override;
  void pow_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) override;
  void matmul_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) override;

  void max_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) override;
  void min_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) override;
  void max_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, std::uint32_t dim, Tensor &gx) override;
  void min_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, std::uint32_t dim, Tensor &gx) override;

  void sum_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) override;
  void logsumexp_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) override;
  void broadcast_fw_impl(const Tensor &x, std::uint32_t dim, std::uint32_t size, Tensor &y) override;

  void batch_pick_fw_impl(const Tensor &x, const std::vector<std::uint32_t> &ids, Tensor &y) override;
  void batch_slice_fw_impl(const Tensor &x, std::uint32_t offset, Tensor &y) override;
  void batch_concat_fw_impl(const std::vector<const Tensor *> &xs, Tensor &y) override;
  void batch_sum_fw_impl(const Tensor &x, Tensor &y) override;

  void batch_pick_bw_impl(const Tensor &gy, const std::vector<std::uint32_t> &ids, Tensor &gx) override;
  void batch_slice_bw_impl(const Tensor &gy, std::uint32_t offset, Tensor &gx) override;

  void conv2d_fw_impl(const Tensor &x, const Tensor &w,
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1,
      std::uint32_t dilation0, std::uint32_t dilation1,
      Tensor &y) override;

  void conv2d_bw_impl(
      const Tensor &x, const Tensor &w, const Tensor &y, const Tensor &gy,
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1,
      std::uint32_t dilation0, std::uint32_t dilation1,
      Tensor &gx, Tensor &gw) override;

  void max_pool2d_fw_impl(
      const Tensor &x,
      std::uint32_t window0, std::uint32_t window1,
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1,
      Tensor &y) override;

  void max_pool2d_bw_impl(
      const Tensor &x, const Tensor &y, const Tensor &gy,
      std::uint32_t window0, std::uint32_t window1,
      std::uint32_t padding0, std::uint32_t padding1,
      std::uint32_t stride0, std::uint32_t stride1,
      Tensor &gx) override;

  void inplace_multiply_const_impl(float k, Tensor &x) override;

  void inplace_add_impl(const Tensor &x, Tensor &y) override;
  void inplace_subtract_impl(const Tensor &x, Tensor &y) override;

private:
  DefaultRandomizer randomizer_;
};

}  // namespace devices
}  // namespace primitiv

#endif  // PRIMITIV_DEVICES_NAIVE_DEVICE_H_
