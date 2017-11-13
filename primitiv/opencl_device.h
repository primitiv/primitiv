#ifndef PRIMITIV_OPENCL_DEVICE_H_
#define PRIMITIV_OPENCL_DEVICE_H_

#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <primitiv/device.h>

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
  static std::uint32_t num_devices();

//   explicit OpenCL(std::uint32_t platform_id, std::uint32_t device_id);
  OpenCL(std::uint32_t platform_id, std::uint32_t device_id);

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

  void random_bernoulli_impl(float p, Tensor &y) override { THROW_ERROR("not implemented"); }
  void random_uniform_impl(float lower, float upper, Tensor &y) override { THROW_ERROR("not implemented"); }
  void random_normal_impl(float mean, float sd, Tensor &y) override { THROW_ERROR("not implemented"); }
  void random_log_normal_impl(float mean, float sd, Tensor &y) override { THROW_ERROR("not implemented"); }

  void pick_fw_impl(const Tensor &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim, Tensor &y) override;
  void slice_fw_impl(const Tensor &x, std::uint32_t dim, std::uint32_t offset, Tensor &y) override;
  void concat_fw_impl(const std::vector<const Tensor *> &xs, std::uint32_t dim, Tensor &y) override;

  void pick_bw_impl(const Tensor &gy, const std::vector<std::uint32_t> &ids, std::uint32_t dim, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void slice_bw_impl(const Tensor &gy, std::uint32_t dim, std::uint32_t offset, Tensor &gx) override { THROW_ERROR("not implemented"); }

  void negate_fw_impl(const Tensor &x, Tensor &y) override { THROW_ERROR("not implemented"); }
  void sqrt_fw_impl(const Tensor &x, Tensor &y) override { THROW_ERROR("not implemented"); }
  void exp_fw_impl(const Tensor &x, Tensor &y) override { THROW_ERROR("not implemented"); }
  void log_fw_impl(const Tensor &x, Tensor &y) override { THROW_ERROR("not implemented"); }
  void tanh_fw_impl(const Tensor &x, Tensor &y) override { THROW_ERROR("not implemented"); }
  void sigmoid_fw_impl(const Tensor &x, Tensor &y) override { THROW_ERROR("not implemented"); }
  void softplus_fw_impl(const Tensor &x, Tensor &y) override { THROW_ERROR("not implemented"); }
  void sin_fw_impl(const Tensor &x, Tensor &y) override { THROW_ERROR("not implemented"); }
  void cos_fw_impl(const Tensor &x, Tensor &y) override { THROW_ERROR("not implemented"); }
  void tan_fw_impl(const Tensor &x, Tensor &y) override { THROW_ERROR("not implemented"); }
  void transpose_fw_impl(const Tensor &x, Tensor &y) override { THROW_ERROR("not implemented"); }

  void sqrt_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void exp_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void log_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void tanh_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void sigmoid_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void softplus_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void sin_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void cos_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void tan_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void transpose_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) override { THROW_ERROR("not implemented"); }

  void add_const_fw_impl(const Tensor &x, float k, Tensor &y) override { THROW_ERROR("not implemented"); }
  void subtract_const_r_fw_impl(const Tensor &x, float k, Tensor &y) override { THROW_ERROR("not implemented"); }
  void subtract_const_l_fw_impl(const Tensor &x, float k, Tensor &y) override { THROW_ERROR("not implemented"); }
  void multiply_const_fw_impl(const Tensor &x, float k, Tensor &y) override { THROW_ERROR("not implemented"); }
  void divide_const_r_fw_impl(const Tensor &x, float k, Tensor &y) override { THROW_ERROR("not implemented"); }
  void divide_const_l_fw_impl(const Tensor &x, float k, Tensor &y) override { THROW_ERROR("not implemented"); }
  void prelu_fw_impl(const Tensor &x, float k, Tensor &y) override { THROW_ERROR("not implemented"); }
  void elu_fw_impl(const Tensor &x, float k, Tensor &y) override { THROW_ERROR("not implemented"); }

  void add_const_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void subtract_const_r_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void subtract_const_l_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void multiply_const_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void divide_const_r_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void divide_const_l_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void prelu_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override { THROW_ERROR("not implemented"); }
  void elu_bw_impl(const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) override { THROW_ERROR("not implemented"); }

  void add_scalar_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override { THROW_ERROR("not implemented"); }
  void subtract_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override { THROW_ERROR("not implemented"); }
  void subtract_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override { THROW_ERROR("not implemented"); }
  void multiply_scalar_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override { THROW_ERROR("not implemented"); }
  void divide_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override { THROW_ERROR("not implemented"); }
  void divide_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) override { THROW_ERROR("not implemented"); }

  void add_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override { THROW_ERROR("not implemented"); }
  void subtract_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override { THROW_ERROR("not implemented"); }
  void multiply_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override { THROW_ERROR("not implemented"); }
  void divide_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override { THROW_ERROR("not implemented"); }
  void matmul_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) override { THROW_ERROR("not implemented"); }

  void add_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) override { THROW_ERROR("not implemented"); }
  void subtract_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) override { THROW_ERROR("not implemented"); }
  void multiply_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) override { THROW_ERROR("not implemented"); }
  void divide_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) override { THROW_ERROR("not implemented"); }
  void matmul_bw_impl(
      const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
      Tensor &ga, Tensor &gb) override { THROW_ERROR("not implemented"); }

  void sum_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) override;
  void logsumexp_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) override { THROW_ERROR("not implemented"); }
  void broadcast_fw_impl(const Tensor &x, std::uint32_t dim, std::uint32_t size, Tensor &y) override { THROW_ERROR("not implemented"); }
  void batch_sum_fw_impl(const Tensor &x, Tensor &y) override { THROW_ERROR("not implemented"); }

  void inplace_multiply_const_impl(float k, Tensor &x) override { THROW_ERROR("not implemented"); }

  void inplace_add_impl(const Tensor &x, Tensor &y) override { THROW_ERROR("not implemented"); }
  void inplace_subtract_impl(const Tensor &x, Tensor &y) override { THROW_ERROR("not implemented"); }

private:
  cl::Device device_;
  cl::Context context_;
  std::uint32_t dev_id_;
  std::uint32_t dim1_x_;

  std::array<cl::Kernel, 11> argmax_kernel_;
  std::array<cl::Kernel, 11> argmin_kernel_;
  cl::Kernel set_identity_kernel_;
  cl::Kernel pick_fw_kernel_;
  cl::Kernel slice_fw_kernel_;
  cl::Kernel concat_fw_kernel_;
  std::array<cl::Kernel, 11> sum_fw_kernel_;
};

}  // namespace devices
}  // namespace primitiv

#endif  // PRIMITIV_CUDA_DEVICE_H_
