#ifndef PRIMITIV_OPENCL_DEVICE_H_
#define PRIMITIV_OPENCL_DEVICE_H_

#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD

#include <CL/cl2.hpp>
#include <primitiv/device.h>
#include <primitiv/random.h>

#include <iostream>

namespace primitiv {
namespace devices {

/**
 * Device class for OpenCL.
 */
class OpenCL : public Device {
  OpenCL() = delete;

public:
  static std::uint32_t num_platforms();
  static std::uint32_t num_devices(std::uint32_t platform_id);

  OpenCL(std::uint32_t platform_id, std::uint32_t device_id);
  OpenCL(std::uint32_t platform_id, std::uint32_t device_id, std::uint32_t rng_seed);

  ~OpenCL() override;

  void dump_description() const override;
  Device::DeviceType type() const override { return Device::DEVICE_TYPE_OPENCL; }

private:
  static std::string kernel_code_generator();

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

  void add_const_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void subtract_const_r_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void subtract_const_l_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void multiply_const_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void divide_const_r_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void divide_const_l_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void prelu_fw_impl(const Tensor &x, float k, Tensor &y) override;
  void elu_fw_impl(const Tensor &x, float k, Tensor &y) override;

  void add_const_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void subtract_const_r_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void subtract_const_l_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void multiply_const_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void divide_const_r_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void divide_const_l_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void prelu_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;
  void elu_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override;

  void add_scalar_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override;
  void subtract_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override;
  void subtract_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override;
  void multiply_scalar_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override;
  void divide_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override;
  void divide_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override;

  void add_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override;
  void subtract_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override;
  void multiply_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override;
  void divide_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override;
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
  void matmul_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) override;

  void sum_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) override;
  void logsumexp_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) override;
  void broadcast_fw_impl(const Tensor &x, std::uint32_t dim, std::uint32_t size, Tensor &y) override;
  void batch_sum_fw_impl(const Tensor &x, Tensor &y) override;

  void inplace_multiply_const_impl(float k, Tensor &x) override;

  void inplace_add_impl(const Tensor &x, Tensor &y) override;
  void inplace_subtract_impl(const Tensor &x, Tensor &y) override;

  /**
   * Internal method to initialize the object.
   */
  void initialize();

private:
  cl::Device device_;
  cl::Context context_;
  std::uint32_t plat_id_;
  std::uint32_t dev_id_;
  DefaultRandomizer randomizer_;

  std::array<cl::Kernel, 11> argmax_kernel_;
  std::uint32_t argmax_kernel_group_size_;
  std::array<cl::Kernel, 11> argmin_kernel_;
  std::uint32_t argmin_kernel_group_size_;

  cl::Kernel set_identity_kernel_;
  std::uint32_t set_identity_kernel_group_size_;

  cl::Kernel pick_fw_kernel_;
  std::uint32_t pick_fw_kernel_group_size_;
  cl::Kernel slice_fw_kernel_;
  std::uint32_t slice_fw_kernel_group_size_;
  cl::Kernel concat_fw_kernel_;
  std::uint32_t concat_fw_kernel_group_size_;

  cl::Kernel pick_bw_kernel_;
  std::uint32_t pick_bw_kernel_group_size_;
  cl::Kernel slice_bw_kernel_;
  std::uint32_t slice_bw_kernel_group_size_;

  cl::Kernel negate_fw_kernel_;
  std::uint32_t negate_fw_kernel_group_size_;
  cl::Kernel sqrt_fw_kernel_;
  std::uint32_t sqrt_fw_kernel_group_size_;
  cl::Kernel exp_fw_kernel_;
  std::uint32_t exp_fw_kernel_group_size_;
  cl::Kernel log_fw_kernel_;
  std::uint32_t log_fw_kernel_group_size_;
  cl::Kernel tanh_fw_kernel_;
  std::uint32_t tanh_fw_kernel_group_size_;
  cl::Kernel sigmoid_fw_kernel_;
  std::uint32_t sigmoid_fw_kernel_group_size_;
  cl::Kernel softplus_fw_kernel_;
  std::uint32_t softplus_fw_kernel_group_size_;
  cl::Kernel sin_fw_kernel_;
  std::uint32_t sin_fw_kernel_group_size_;
  cl::Kernel cos_fw_kernel_;
  std::uint32_t cos_fw_kernel_group_size_;
  cl::Kernel tan_fw_kernel_;
  std::uint32_t tan_fw_kernel_group_size_;
  cl::Kernel transpose_fw_kernel_;
  std::uint32_t transpose_fw_kernel_group_size_;
  std::uint32_t transpose_fw_kernel_group_size_y_;
  std::uint32_t transpose_fw_kernel_group_size_x_;

  cl::Kernel sqrt_bw_kernel_;
  std::uint32_t sqrt_bw_kernel_group_size_;
  cl::Kernel exp_bw_kernel_;
  std::uint32_t exp_bw_kernel_group_size_;
  cl::Kernel log_bw_kernel_;
  std::uint32_t log_bw_kernel_group_size_;
  cl::Kernel tanh_bw_kernel_;
  std::uint32_t tanh_bw_kernel_group_size_;
  cl::Kernel sigmoid_bw_kernel_;
  std::uint32_t sigmoid_bw_kernel_group_size_;
  cl::Kernel softplus_bw_kernel_;
  std::uint32_t softplus_bw_kernel_group_size_;
  cl::Kernel sin_bw_kernel_;
  std::uint32_t sin_bw_kernel_group_size_;
  cl::Kernel cos_bw_kernel_;
  std::uint32_t cos_bw_kernel_group_size_;
  cl::Kernel tan_bw_kernel_;
  std::uint32_t tan_bw_kernel_group_size_;
  cl::Kernel transpose_bw_kernel_;
  std::uint32_t transpose_bw_kernel_group_size_;
  std::uint32_t transpose_bw_kernel_group_size_y_;
  std::uint32_t transpose_bw_kernel_group_size_x_;

  cl::Kernel add_const_fw_kernel_;
  std::uint32_t add_const_fw_kernel_group_size_;
  cl::Kernel subtract_const_r_fw_kernel_;
  std::uint32_t subtract_const_r_fw_kernel_group_size_;
  cl::Kernel subtract_const_l_fw_kernel_;
  std::uint32_t subtract_const_l_fw_kernel_group_size_;
  cl::Kernel multiply_const_fw_kernel_;
  std::uint32_t multiply_const_fw_kernel_group_size_;
  cl::Kernel divide_const_r_fw_kernel_;
  std::uint32_t divide_const_r_fw_kernel_group_size_;
  cl::Kernel divide_const_l_fw_kernel_;
  std::uint32_t divide_const_l_fw_kernel_group_size_;
  cl::Kernel prelu_fw_kernel_;
  std::uint32_t prelu_fw_kernel_group_size_;
  cl::Kernel elu_fw_kernel_;
  std::uint32_t elu_fw_kernel_group_size_;

  cl::Kernel add_const_bw_kernel_;
  std::uint32_t add_const_bw_kernel_group_size_;
  cl::Kernel subtract_const_r_bw_kernel_;
  std::uint32_t subtract_const_r_bw_kernel_group_size_;
  cl::Kernel subtract_const_l_bw_kernel_;
  std::uint32_t subtract_const_l_bw_kernel_group_size_;
  cl::Kernel multiply_const_bw_kernel_;
  std::uint32_t multiply_const_bw_kernel_group_size_;
  cl::Kernel divide_const_r_bw_kernel_;
  std::uint32_t divide_const_r_bw_kernel_group_size_;
  cl::Kernel divide_const_l_bw_kernel_;
  std::uint32_t divide_const_l_bw_kernel_group_size_;
  cl::Kernel prelu_bw_kernel_;
  std::uint32_t prelu_bw_kernel_group_size_;
  cl::Kernel elu_bw_kernel_;
  std::uint32_t elu_bw_kernel_group_size_;

  cl::Kernel add_scalar_fw_kernel_;
  std::uint32_t add_scalar_fw_kernel_group_size_;
  cl::Kernel subtract_scalar_r_fw_kernel_;
  std::uint32_t subtract_scalar_r_fw_kernel_group_size_;
  cl::Kernel subtract_scalar_l_fw_kernel_;
  std::uint32_t subtract_scalar_l_fw_kernel_group_size_;
  cl::Kernel multiply_scalar_fw_kernel_;
  std::uint32_t multiply_scalar_fw_kernel_group_size_;
  cl::Kernel divide_scalar_r_fw_kernel_;
  std::uint32_t divide_scalar_r_fw_kernel_group_size_;
  cl::Kernel divide_scalar_l_fw_kernel_;
  std::uint32_t divide_scalar_l_fw_kernel_group_size_;

  cl::Kernel add_fw_kernel_;
  std::uint32_t add_fw_kernel_group_size_;
  cl::Kernel subtract_fw_kernel_;
  std::uint32_t subtract_fw_kernel_group_size_;
  cl::Kernel multiply_fw_kernel_;
  std::uint32_t multiply_fw_kernel_group_size_;
  cl::Kernel divide_fw_kernel_;
  std::uint32_t divide_fw_kernel_group_size_;

  cl::Kernel add_bw_kernel_;
  std::uint32_t add_bw_kernel_group_size_;
  cl::Kernel subtract_bw_kernel_;
  std::uint32_t subtract_bw_kernel_group_size_;
  cl::Kernel multiply_bw_kernel_;
  std::uint32_t multiply_bw_kernel_group_size_;
  cl::Kernel divide_bw_kernel_;
  std::uint32_t divide_bw_kernel_group_size_;

  std::array<cl::Kernel, 11> sum_fw_kernel_;
  std::uint32_t sum_fw_kernel_group_size_;
  std::array<cl::Kernel, 11> logsumexp_fw_kernel_;
  std::uint32_t logsumexp_fw_kernel_group_size_;

  cl::Kernel broadcast_fw_kernel_;
  std::uint32_t broadcast_fw_kernel_group_size_;
  cl::Kernel batch_sum_fw_kernel_;
  std::uint32_t batch_sum_fw_kernel_group_size_;

  cl::Kernel inplace_multiply_const_kernel_;
  std::uint32_t inplace_multiply_const_kernel_group_size_;

  cl::Kernel inplace_add_kernel_;
  std::uint32_t inplace_add_kernel_group_size_;
  cl::Kernel inplace_subtract_kernel_;
  std::uint32_t inplace_subtract_kernel_group_size_;
};

}  // namespace devices
}  // namespace primitiv

#endif  // PRIMITIV_CUDA_DEVICE_H_
