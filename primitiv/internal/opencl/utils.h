#ifndef PRIMITIV_INTERNAL_OPENCL_UTILS_H_
#define PRIMITIV_INTERNAL_OPENCL_UTILS_H_

#include <primitiv/config.h>

#include <random>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_TARGET_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#include <CL/cl2.hpp>

#include <primitiv/core/error.h>
#include <primitiv/core/memory_pool.h>
#include <primitiv/core/random.h>
#include <primitiv/devices/opencl/device.h>

namespace {

/**
 * Generates source code of all kernel functions.
 * @return Source code of kernel functions.
 */
std::string generate_kernels() {
  return {
    // `kernels.inc` is generated from `kernels.cl`
#include "primitiv/devices/opencl/kernels.inc"
  };
}

/**
 * Returns the list of available platforms.
 * @return List of available cl::Platform.
 */
std::vector<cl::Platform> get_all_platforms() {
  std::vector<cl::Platform> ret;
  cl::Platform::get(&ret);
  return ret;
}

/**
 * Returns the list of available devices on the specified platform.
 * @param platform_id Platform ID.
 * @return List of available cl::Device.
 */
std::vector<cl::Device> get_all_devices(std::uint32_t platform_id) {
  const auto all_pfs = ::get_all_platforms();
  if (platform_id >= all_pfs.size()) {
    PRIMITIV_THROW_ERROR("Invalid platform ID: " << platform_id);
  }
  std::vector<cl::Device> ret;
  all_pfs[platform_id].getDevices(CL_DEVICE_TYPE_ALL, &ret);
  return ret;
}

/**
 * Returns the cl::Device corresponding to the specified IDs.
 * @param platform_id Platform ID.
 * @param device_id Device ID.
 * @return Corresponding cl::Device object.
 */
cl::Device get_device(std::uint32_t platform_id, std::uint32_t device_id) {
  const auto all_devs = ::get_all_devices(platform_id);
  if (device_id >= all_devs.size()) {
    PRIMITIV_THROW_ERROR(
        "Invalid device ID: " << device_id
        << " (on the platform " << platform_id << ")");
  }
  return all_devs[device_id];
}

}  // namespace

namespace primitiv {
namespace devices {

/**
 * Hidden objects of OpenCL devices.
 */
struct OpenCLInternalState {
private:
  /**
   * aHelper to obtain maximum work group size of the kernel.
   */
  std::size_t get_work_group_size(const cl::Kernel &kernel) {
    return kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);
  }

  /**
   * Helper to find an integer x that satisfy:
   * 1. x == 2^n
   * 2. x <= size
   */
  std::uint32_t calc_dim1_size(std::uint32_t size) {
    std::uint32_t ret = 1;
    while (ret << 1 <= size) ret <<= 1;
    return ret;
  }

  /**
   * Helper to find two sizes (x, y) that satisfy:
   * 1.x == 2^n, y == 2^m
   * 2. x * y <= size
   * 3. x / y == 1 or 2
   */
  void calc_dim2_sizes(std::uint32_t size, std::uint32_t &x, std::uint32_t &y) {
    x = y = 1;
    bool p = true;
    while ((x * y) << 1 <= size) {
      (p ? x : y) <<= 1;
      p = !p;
    }
  }

public:
  OpenCLInternalState(
      std::uint32_t pf_id, std::uint32_t dev_id, std::uint32_t rng_seed)
    : randomizer_(rng_seed)
    , device(::get_device(pf_id, dev_id))
    , context({ device })
    , queue(context, device, 0)
    , pool(
        [this](std::size_t size) -> void * {  // allocator
          return static_cast<void *>(
              new cl::Buffer(
                context,
                CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                size,
                nullptr));
        },
        [this](void *ptr) -> void {  // deleter
          // NOTE(odashi):
          // Deleting cl::Buffer does NOT block the process regardless whether
          // the remaining kernel functions are still working or not.
          // We have to manually wait for finishing all kernel functions to
          // prevent memory corruption.
          queue.finish();
          // Then, we can delete the buffer safely.
          delete static_cast<cl::Buffer *>(ptr);
        }) {
      cl::Program program(context, ::generate_kernels());
      try {
        program.build({device});
      } catch (...) {
        PRIMITIV_THROW_ERROR("OpenCL kernel compile error:" << std::endl << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
      }

#define CONFIGURE_KERNEL(name) \
      { \
        name##_kernel = cl::Kernel(program, #name "_kernel"); \
        name##_group_size = get_work_group_size(name##_kernel); \
      }

#define CONFIGURE_KERNEL_LIST(name) \
      { \
        for (std::uint32_t i = 0; i <= 10; ++i) { \
          std::ostringstream ss; \
          ss << #name "_kernel_" << (1 << i); \
          name##_kernel[i] = cl::Kernel(program, ss.str().c_str()); \
        } \
        name##_group_size = get_work_group_size(name##_kernel[0]); \
      }

      CONFIGURE_KERNEL_LIST(argmax);
      CONFIGURE_KERNEL_LIST(argmin);
      argmax_group_size = calc_dim1_size(argmax_group_size);
      argmin_group_size = calc_dim1_size(argmin_group_size);

      CONFIGURE_KERNEL(set_identity);

      CONFIGURE_KERNEL(pick_fw);
      CONFIGURE_KERNEL(slice_fw);
      CONFIGURE_KERNEL(concat_fw);

      CONFIGURE_KERNEL(pick_bw);
      CONFIGURE_KERNEL(slice_bw);

      CONFIGURE_KERNEL(negate_fw);
      CONFIGURE_KERNEL(abs_fw);
      CONFIGURE_KERNEL(sqrt_fw);
      CONFIGURE_KERNEL(exp_fw);
      CONFIGURE_KERNEL(log_fw);
      CONFIGURE_KERNEL(tanh_fw);
      CONFIGURE_KERNEL(sigmoid_fw);
      CONFIGURE_KERNEL(softplus_fw);
      CONFIGURE_KERNEL(sin_fw);
      CONFIGURE_KERNEL(cos_fw);
      CONFIGURE_KERNEL(tan_fw);

      CONFIGURE_KERNEL(abs_bw);
      CONFIGURE_KERNEL(sqrt_bw);
      CONFIGURE_KERNEL(exp_bw);
      CONFIGURE_KERNEL(log_bw);
      CONFIGURE_KERNEL(tanh_bw);
      CONFIGURE_KERNEL(sigmoid_bw);
      CONFIGURE_KERNEL(softplus_bw);
      CONFIGURE_KERNEL(sin_bw);
      CONFIGURE_KERNEL(cos_bw);
      CONFIGURE_KERNEL(tan_bw);

      CONFIGURE_KERNEL(transpose_fw);
      CONFIGURE_KERNEL(transpose_bw);

      calc_dim2_sizes(
          transpose_fw_group_size,
          transpose_fw_group_size_x, transpose_fw_group_size_y);
      calc_dim2_sizes(
          transpose_bw_group_size,
          transpose_bw_group_size_x, transpose_bw_group_size_y);

      CONFIGURE_KERNEL(flip_fw);
      CONFIGURE_KERNEL(flip_bw);

      calc_dim2_sizes(
          flip_fw_group_size,
          flip_fw_group_size_x, flip_fw_group_size_y);
      calc_dim2_sizes(
          flip_bw_group_size,
          flip_bw_group_size_x, flip_bw_group_size_y);

      CONFIGURE_KERNEL(permute_dims_fw);
      CONFIGURE_KERNEL(permute_dims_bw);

      CONFIGURE_KERNEL(add_const_fw);
      CONFIGURE_KERNEL(subtract_const_r_fw);
      CONFIGURE_KERNEL(subtract_const_l_fw);
      CONFIGURE_KERNEL(multiply_const_fw);
      CONFIGURE_KERNEL(divide_const_r_fw);
      CONFIGURE_KERNEL(divide_const_l_fw);
      CONFIGURE_KERNEL(pow_const_r_fw);
      CONFIGURE_KERNEL(pow_const_l_fw);
      CONFIGURE_KERNEL(prelu_fw);
      CONFIGURE_KERNEL(elu_fw);

      CONFIGURE_KERNEL(pown_fw);

      CONFIGURE_KERNEL(add_const_bw);
      CONFIGURE_KERNEL(subtract_const_r_bw);
      CONFIGURE_KERNEL(subtract_const_l_bw);
      CONFIGURE_KERNEL(multiply_const_bw);
      CONFIGURE_KERNEL(divide_const_r_bw);
      CONFIGURE_KERNEL(divide_const_l_bw);
      CONFIGURE_KERNEL(pow_const_r_bw);
      CONFIGURE_KERNEL(pow_const_l_bw);
      CONFIGURE_KERNEL(prelu_bw);
      CONFIGURE_KERNEL(elu_bw);

      CONFIGURE_KERNEL(pown_bw);

      CONFIGURE_KERNEL(add_scalar_fw);
      CONFIGURE_KERNEL(subtract_scalar_r_fw);
      CONFIGURE_KERNEL(subtract_scalar_l_fw);
      CONFIGURE_KERNEL(multiply_scalar_fw);
      CONFIGURE_KERNEL(divide_scalar_r_fw);
      CONFIGURE_KERNEL(divide_scalar_l_fw);
      CONFIGURE_KERNEL(pow_scalar_r_fw);
      CONFIGURE_KERNEL(pow_scalar_l_fw);

      CONFIGURE_KERNEL(add_fw);
      CONFIGURE_KERNEL(subtract_fw);
      CONFIGURE_KERNEL(multiply_fw);
      CONFIGURE_KERNEL(divide_fw);
      CONFIGURE_KERNEL(pow_fw);

      CONFIGURE_KERNEL(add_bw);
      CONFIGURE_KERNEL(subtract_bw);
      CONFIGURE_KERNEL(multiply_bw);
      CONFIGURE_KERNEL(divide_bw);
      CONFIGURE_KERNEL(pow_bw);

      CONFIGURE_KERNEL_LIST(max_fw);
      CONFIGURE_KERNEL_LIST(min_fw);
      CONFIGURE_KERNEL_LIST(max_bw);
      CONFIGURE_KERNEL_LIST(min_bw);
      max_fw_group_size = calc_dim1_size(max_fw_group_size);
      min_fw_group_size = calc_dim1_size(min_fw_group_size);
      max_bw_group_size = calc_dim1_size(max_bw_group_size);
      min_bw_group_size = calc_dim1_size(min_bw_group_size);

      CONFIGURE_KERNEL_LIST(sum_fw);
      CONFIGURE_KERNEL_LIST(logsumexp_fw);
      sum_fw_group_size = calc_dim1_size(sum_fw_group_size);
      logsumexp_fw_group_size = calc_dim1_size(logsumexp_fw_group_size);

      CONFIGURE_KERNEL(broadcast_fw);
      CONFIGURE_KERNEL(batch_pick_fw);
      CONFIGURE_KERNEL(batch_slice_fw);
      CONFIGURE_KERNEL(batch_concat_fw);
      CONFIGURE_KERNEL(batch_sum_fw);

      CONFIGURE_KERNEL(batch_pick_bw);
      CONFIGURE_KERNEL(batch_slice_bw);

      CONFIGURE_KERNEL(inplace_multiply_const);
      CONFIGURE_KERNEL(inplace_add);
      CONFIGURE_KERNEL(inplace_subtract);

#undef CONFIGURE_KERNEL
#undef CONFIGURE_KERNEL_LIST
    }

  DefaultRandomizer randomizer_;
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;
  MemoryPool pool;

#define DECL_KERNEL(name) \
  cl::Kernel name##_kernel; \
  std::uint32_t name##_group_size;
#define DECL_KERNEL_LIST(name, size) \
  std::array<cl::Kernel, size> name##_kernel; \
  std::uint32_t name##_group_size;

  DECL_KERNEL_LIST(argmax, 11);
  DECL_KERNEL_LIST(argmin, 11);

  DECL_KERNEL(set_identity);

  DECL_KERNEL(pick_fw);
  DECL_KERNEL(slice_fw);
  DECL_KERNEL(concat_fw);

  DECL_KERNEL(pick_bw);
  DECL_KERNEL(slice_bw);

  DECL_KERNEL(negate_fw);
  DECL_KERNEL(abs_fw);
  DECL_KERNEL(sqrt_fw);
  DECL_KERNEL(exp_fw);
  DECL_KERNEL(log_fw);
  DECL_KERNEL(tanh_fw);
  DECL_KERNEL(sigmoid_fw);
  DECL_KERNEL(softplus_fw);
  DECL_KERNEL(sin_fw);
  DECL_KERNEL(cos_fw);
  DECL_KERNEL(tan_fw);

  DECL_KERNEL(transpose_fw);
  std::uint32_t transpose_fw_group_size_x;
  std::uint32_t transpose_fw_group_size_y;
  DECL_KERNEL(permute_dims_fw);

  DECL_KERNEL(flip_fw);
  std::uint32_t flip_fw_group_size_x;
  std::uint32_t flip_fw_group_size_y;

  DECL_KERNEL(abs_bw);
  DECL_KERNEL(sqrt_bw);
  DECL_KERNEL(exp_bw);
  DECL_KERNEL(log_bw);
  DECL_KERNEL(tanh_bw);
  DECL_KERNEL(sigmoid_bw);
  DECL_KERNEL(softplus_bw);
  DECL_KERNEL(sin_bw);
  DECL_KERNEL(cos_bw);
  DECL_KERNEL(tan_bw);

  DECL_KERNEL(transpose_bw);
  std::uint32_t transpose_bw_group_size_x;
  std::uint32_t transpose_bw_group_size_y;
  DECL_KERNEL(permute_dims_bw);

  DECL_KERNEL(flip_bw);
  std::uint32_t flip_bw_group_size_x;
  std::uint32_t flip_bw_group_size_y;

  DECL_KERNEL(add_const_fw);
  DECL_KERNEL(subtract_const_r_fw);
  DECL_KERNEL(subtract_const_l_fw);
  DECL_KERNEL(multiply_const_fw);
  DECL_KERNEL(divide_const_r_fw);
  DECL_KERNEL(divide_const_l_fw);
  DECL_KERNEL(pow_const_r_fw);
  DECL_KERNEL(pow_const_l_fw);
  DECL_KERNEL(prelu_fw);
  DECL_KERNEL(elu_fw);

  DECL_KERNEL(pown_fw);

  DECL_KERNEL(add_const_bw);
  DECL_KERNEL(subtract_const_r_bw);
  DECL_KERNEL(subtract_const_l_bw);
  DECL_KERNEL(multiply_const_bw);
  DECL_KERNEL(divide_const_r_bw);
  DECL_KERNEL(divide_const_l_bw);
  DECL_KERNEL(pow_const_r_bw);
  DECL_KERNEL(pow_const_l_bw);
  DECL_KERNEL(prelu_bw);
  DECL_KERNEL(elu_bw);

  DECL_KERNEL(pown_bw);

  DECL_KERNEL(add_scalar_fw);
  DECL_KERNEL(subtract_scalar_r_fw);
  DECL_KERNEL(subtract_scalar_l_fw);
  DECL_KERNEL(multiply_scalar_fw);
  DECL_KERNEL(divide_scalar_r_fw);
  DECL_KERNEL(divide_scalar_l_fw);
  DECL_KERNEL(pow_scalar_r_fw);
  DECL_KERNEL(pow_scalar_l_fw);

  DECL_KERNEL(add_fw);
  DECL_KERNEL(subtract_fw);
  DECL_KERNEL(multiply_fw);
  DECL_KERNEL(divide_fw);
  DECL_KERNEL(pow_fw);

  DECL_KERNEL(add_bw);
  DECL_KERNEL(subtract_bw);
  DECL_KERNEL(multiply_bw);
  DECL_KERNEL(divide_bw);
  DECL_KERNEL(pow_bw);

  DECL_KERNEL_LIST(max_fw, 11);
  DECL_KERNEL_LIST(min_fw, 11);
  DECL_KERNEL_LIST(max_bw, 11);
  DECL_KERNEL_LIST(min_bw, 11);

  DECL_KERNEL_LIST(sum_fw, 11);
  DECL_KERNEL_LIST(logsumexp_fw, 11);

  DECL_KERNEL(broadcast_fw);
  DECL_KERNEL(batch_pick_fw);
  DECL_KERNEL(batch_slice_fw);
  DECL_KERNEL(batch_concat_fw);
  DECL_KERNEL(batch_sum_fw);

  DECL_KERNEL(batch_pick_bw);
  DECL_KERNEL(batch_slice_bw);

  DECL_KERNEL(inplace_multiply_const);
  DECL_KERNEL(inplace_add);
  DECL_KERNEL(inplace_subtract);

#undef DECL_KERNEL
#undef DECL_KERNEL_LIST
};

}  // namespace cuda
}  // namespace primitiv

#endif  // PRIMITIV_INTERNAL_OPENCL_UTILS_H_
