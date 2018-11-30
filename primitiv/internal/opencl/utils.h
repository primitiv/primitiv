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
class OpenCLKernel {
public:
  OpenCLKernel() {}

  bool initialized() { return kernel_() != nullptr; }

  cl::Kernel &kernel() {
    return kernel_;
  }

  cl::detail::size_t_array &group_size() {
    return group_size_;
  }

private:
  cl::Kernel kernel_;
  cl::detail::size_t_array group_size_;
};

struct OpenCLInternalState {
private:
  /**
   * aHelper to obtain maximum work group size of the kernel.
   */
  cl::detail::size_t_array reqd_work_group_size(const cl::Kernel &kernel) {
    return kernel.getWorkGroupInfo<CL_KERNEL_COMPILE_WORK_GROUP_SIZE>(device);
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
    }

  void initialize_kernel(OpenCLKernel &kernel, std::string source, const char* name) {
    cl::Program program(context, source);
    try {
      program.build({ device });
    } catch (...) {
      PRIMITIV_THROW_ERROR(
          "OpenCL kernel compile error:" << std::endl
          << "Function: " << name << std::endl
          << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
    }
    kernel.kernel() = cl::Kernel(program, name);
    kernel.group_size() = reqd_work_group_size(kernel.kernel());
  }

  DefaultRandomizer randomizer_;
  cl::Device device;
  cl::Context context;
  cl::CommandQueue queue;
  MemoryPool pool;

#define DECL_KERNEL(name) \
  OpenCLKernel name##_kernel;

  DECL_KERNEL(argmax);
  DECL_KERNEL(argmin);

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
  DECL_KERNEL(permute_dims_fw);

  DECL_KERNEL(flip_fw);

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
  DECL_KERNEL(permute_dims_bw);

  DECL_KERNEL(flip_bw);

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

  DECL_KERNEL(max_fw);
  DECL_KERNEL(min_fw);
  DECL_KERNEL(max_bw);
  DECL_KERNEL(min_bw);

  DECL_KERNEL(sum_fw);
  DECL_KERNEL(logsumexp_fw);

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
};

}  // namespace cuda
}  // namespace primitiv

#endif  // PRIMITIV_INTERNAL_OPENCL_UTILS_H_
