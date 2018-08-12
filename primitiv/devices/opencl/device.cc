#include <primitiv/config.h>

#include <algorithm>
#include <iostream>
#include <random>

#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_CL_1_2_DEFAULT_BUILD
#include <CL/cl2.hpp>
#include <clblast.h>

#include <primitiv/core/error.h>
#include <primitiv/core/memory_pool.h>
#include <primitiv/core/random.h>
#include <primitiv/devices/opencl/device.h>

namespace {

/**
 * Copies a device buffer to a host array.
 * @param queue cl::CommandQueue object to perform operations.
 * @param buffer cl::Buffer object to be updated.
 * @param data Array of the data.
 * @param size Number of objects in `data`.
 */
template<typename T>
void read_buffer(
    cl::CommandQueue &queue, const cl::Buffer &buffer,
    T data[], std::size_t size) {
  queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(T) * size, data);
}

/**
 * Copies a host array to a device buffer.
 * @param queue cl::CommandQueue object to perform operations.
 * @param buffer cl::Buffer object to be updated.
 * @param data Array of the data.
 * @param size Number of objects in `data`.
 */
template<typename T>
void write_buffer(
    cl::CommandQueue &queue, cl::Buffer &buffer,
    const T data[], std::size_t size) {
  // NOTE(odashi):
  // Some devices could not directly write their buffers.
  // (I observed this issue using Intel GPUs.)
  // For now, we disabled below code,
  //
  //queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, sizeof(T) * size, data);
  //
  // and enables copying through memory mapping.
  T *mapped_ptr = static_cast<T *>(
      queue.enqueueMapBuffer(
        buffer, CL_TRUE, CL_MAP_WRITE, 0, sizeof(T) * size, 0));
  std::memcpy(mapped_ptr, data, sizeof(T) * size);
  queue.enqueueUnmapMemObject(buffer, mapped_ptr);
}

/**
 * Obtains mutable cl::Buffer from shared_ptr<void>.
 * @param ptr Target shared_ptr object.
 * @return cl::Buffer object which the shared_ptr holds.
 */
cl::Buffer &get_buffer(std::shared_ptr<void> &ptr) {
  return *static_cast<cl::Buffer *>(ptr.get());
}

/**
 * Obtains the number of blocks in one parallel operation.
 * @param size Total number of threads.
 * @param num_threads Number of threads in one block.
 */
std::uint32_t calc_num_blocks(std::uint32_t size, std::uint32_t num_threads) {
  return (size + num_threads - 1) / num_threads;
}

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

#define CDATA(x) (*static_cast<const cl::Buffer *>(get_handle(x)))
#define MDATA(x) (*static_cast<cl::Buffer *>(get_mutable_handle(x)))

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

std::uint32_t OpenCL::num_platforms() {
  return ::get_all_platforms().size();
}

std::uint32_t OpenCL::num_devices(std::uint32_t platform_id) {
  return ::get_all_devices(platform_id).size();
}

void OpenCL::assert_support(
    std::uint32_t platform_id, std::uint32_t device_id) {
  const cl::Device dev = ::get_device(platform_id, device_id);

  // Checks whether the device is globally available.
  if (!dev.getInfo<CL_DEVICE_AVAILABLE>()) {
    PRIMITIV_THROW_ERROR(
        "OpenCL Device " << device_id << " on the platform " << platform_id
        << " is not available (CL_DEVICE_AVAILABLE == false).");
  }

  // Checks other minimum requirements.
#define CHECK_REQUIREMENT(name, value) \
  { \
    const auto actual = dev.getInfo<name>(); \
    if (actual < (value)) { \
      PRIMITIV_THROW_ERROR( \
          "OpenCL Device " << device_id << " on the platform " << platform_id \
          << " does not satisfy the minimum requirement by primitiv. " \
          << "property: " << #name << ", " \
          << "value: " << actual << ", " \
          << "required at least: " << (value)); \
    } \
  }
#define CHECK_REQUIREMENT_VECTOR(name, index, value) \
  { \
    const auto actual = dev.getInfo<name>()[index]; \
    if (actual < (value)) { \
      PRIMITIV_THROW_ERROR( \
          "OpenCL Device " << device_id << " on the platform " << platform_id \
          << " does not satisfy the minimum requirement by primitiv. " \
          << "property: " << #name << "[" << #index << "], " \
          << "value: " << actual << ", " \
          << "required at least: " << (value)); \
    } \
  } \

  CHECK_REQUIREMENT(CL_DEVICE_GLOBAL_MEM_SIZE, 1ull * (1ull << 30));
  CHECK_REQUIREMENT(CL_DEVICE_LOCAL_MEM_SIZE, 16ull * (1ull << 10));
  CHECK_REQUIREMENT(CL_DEVICE_MAX_WORK_GROUP_SIZE, 256);
  CHECK_REQUIREMENT_VECTOR(CL_DEVICE_MAX_WORK_ITEM_SIZES, 0, 256);
  CHECK_REQUIREMENT_VECTOR(CL_DEVICE_MAX_WORK_ITEM_SIZES, 1, 16);
  CHECK_REQUIREMENT_VECTOR(CL_DEVICE_MAX_WORK_ITEM_SIZES, 2, 1);
  // NOTE(odashi): OpenCL does not support explicit grid sizes.

#undef CHECK_REQUIREMENT
#undef CHECK_REQUIREMENT_VECTOR
}

void OpenCL::initialize() {
  assert_support(pf_id_, dev_id_);
  state_.reset(new OpenCLInternalState(pf_id_, dev_id_, rng_seed_));
}

OpenCL::OpenCL(std::uint32_t platform_id, std::uint32_t device_id)
: pf_id_(platform_id)
, dev_id_(device_id)
, rng_seed_(std::random_device()()) {
  initialize();
}

OpenCL::OpenCL(
    std::uint32_t platform_id, std::uint32_t device_id, std::uint32_t rng_seed)
: pf_id_(platform_id)
, dev_id_(device_id)
, rng_seed_(rng_seed) {
  initialize();
}

OpenCL::~OpenCL() {
  // Nothing to do for now.
}

void OpenCL::dump_description() const {
  std::cerr << "Device " << this << std::endl;
  std::cerr << "  Type: OpenCL" << std::endl;

  std::cerr << "  Platform ID: " << pf_id_ << std::endl;
  std::cerr << "  Device ID: " << dev_id_ << std::endl;
  std::cerr << "    Vendor ............ "
            << state_->device.getInfo<CL_DEVICE_VENDOR>() << std::endl;
  std::cerr << "    Name .............. "
            << state_->device.getInfo<CL_DEVICE_NAME>() << std::endl;
  std::cerr << "    Global memory ..... "
            << state_->device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() << std::endl;
  std::cerr << "    Local memory ...... "
            << state_->device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() << std::endl;
  std::cerr << "    Work group size ... "
            << state_->device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
  std::cerr << "    Work item size .... ";
  const auto sizes = state_->device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    if (i > 0) std::cerr << ", ";
    std::cerr << sizes[i];
  }
  std::cerr << std::endl;
}

std::shared_ptr<void> OpenCL::new_handle(const Shape &shape) {
  return state_->pool.allocate(sizeof(float) * shape.size());
}

std::vector<float> OpenCL::tensor_to_vector_impl(const Tensor &x) {
  const std::uint32_t size = x.shape().size();
  std::vector<float> ret(size);
  ::read_buffer(state_->queue, CDATA(x), ret.data(), size);
  return ret;
}

std::vector<std::uint32_t> OpenCL::argmax_impl(
    const Tensor &x, std::uint32_t dim) {
  const Shape &shape = x.shape();
  const std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  const std::uint32_t s = shape.lower_volume(dim);
  std::uint32_t group_size = std::min(state_->argmax_group_size, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  std::shared_ptr<void> py = state_->pool.allocate(sizeof(std::uint32_t) * r);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      state_->argmax_kernel[m].setArg(0, CDATA(x)); \
      state_->argmax_kernel[m].setArg(1, s); \
      state_->argmax_kernel[m].setArg(2, n); \
      state_->argmax_kernel[m].setArg(3, ::get_buffer(py)); \
      state_->queue.enqueueNDRangeKernel( \
          state_->argmax_kernel[m], \
          cl::NullRange, cl::NDRange(r * k), cl::NDRange(k)); \
      break;
    CASE(1024, 10);
    CASE(512, 9);
    CASE(256, 8);
    CASE(128, 7);
    CASE(64, 6);
    CASE(32, 5);
    CASE(16, 4);
    CASE(8, 3);
    CASE(4, 2);
    CASE(2, 1);
    CASE(1, 0);
#undef CASE
  }
  std::vector<std::uint32_t> ret(r);
  ::read_buffer(state_->queue, ::get_buffer(py), ret.data(), r);
  return ret;
}

std::vector<std::uint32_t> OpenCL::argmin_impl(
    const Tensor &x, std::uint32_t dim) {
  const Shape &shape = x.shape();
  const std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  const std::uint32_t s = shape.lower_volume(dim);
  std::uint32_t group_size = std::min(state_->argmin_group_size, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  std::shared_ptr<void> py = state_->pool.allocate(sizeof(std::uint32_t) * r);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      state_->argmin_kernel[m].setArg(0, CDATA(x)); \
      state_->argmin_kernel[m].setArg(1, s); \
      state_->argmin_kernel[m].setArg(2, n); \
      state_->argmin_kernel[m].setArg(3, ::get_buffer(py)); \
      state_->queue.enqueueNDRangeKernel( \
          state_->argmin_kernel[m], \
          cl::NullRange, cl::NDRange(r * k), cl::NDRange(k)); \
      break;
    CASE(1024, 10);
    CASE(512, 9);
    CASE(256, 8);
    CASE(128, 7);
    CASE(64, 6);
    CASE(32, 5);
    CASE(16, 4);
    CASE(8, 3);
    CASE(4, 2);
    CASE(2, 1);
    CASE(1, 0);
#undef CASE
  }
  std::vector<std::uint32_t> ret(r);
  ::read_buffer(state_->queue, ::get_buffer(py), ret.data(), r);
  return ret;
}

void OpenCL::reset_tensor_impl(float k, Tensor &x) {
  const std::uint32_t size = x.shape().size();
  state_->queue.enqueueFillBuffer<float>(MDATA(x), k, 0, sizeof(float) * size);
}

void OpenCL::reset_tensor_by_array_impl(const float values[], Tensor &x) {
  const std::uint32_t size = x.shape().size();
  ::write_buffer(state_->queue, MDATA(x), values, size);
}

void OpenCL::copy_tensor_impl(const Tensor &x, Tensor &y) {
  switch (x.device().type()) {
    case DeviceType::NAIVE:
      reset_tensor_by_array(static_cast<const float *>(get_handle(x)), y);
      break;
    case DeviceType::OPENCL:
      if(&x.device() == this) {
        const std::uint32_t size = x.shape().size();
        state_->queue.enqueueCopyBuffer(
            CDATA(x), MDATA(y), 0, 0, sizeof(float) * size);
      } else {
        const std::uint32_t size = x.shape().size();
        cl::CommandQueue &queue_x = static_cast<OpenCL &>(x.device()).state_->queue;
        const float *mapped_ptr_x = static_cast<const float *>(
            queue_x.enqueueMapBuffer(
              CDATA(x), CL_TRUE, CL_MAP_READ, 0, sizeof(float) * size, 0));
        float *mapped_ptr_y = static_cast<float *>(
            state_->queue.enqueueMapBuffer(
              MDATA(y), CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * size, 0));
        std::memcpy(mapped_ptr_y, mapped_ptr_x, sizeof(float) * size);
        queue_x.enqueueUnmapMemObject(
            CDATA(x), const_cast<float *>(mapped_ptr_x));
        state_->queue.enqueueUnmapMemObject(MDATA(y), mapped_ptr_y);
      }
      break;
    default:
      reset_tensor_by_vector(x.to_vector(), y);
  }
}

void OpenCL::identity_impl(Tensor &y) {
  const std::uint32_t size = y.shape().size();
  const std::uint32_t skip = y.shape()[0] + 1;
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->set_identity_group_size);
  state_->set_identity_kernel.setArg(0, size);
  state_->set_identity_kernel.setArg(1, skip);
  state_->set_identity_kernel.setArg(2, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->set_identity_kernel, cl::NullRange,
      cl::NDRange(num_blocks * state_->set_identity_group_size),
      cl::NDRange(state_->set_identity_group_size));
}

void OpenCL::random_bernoulli_impl(float p, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  float *mapped_ptr = static_cast<float *>(
      state_->queue.enqueueMapBuffer(
        MDATA(y), CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * size, 0));
  state_->randomizer_.fill_bernoulli(p, size, mapped_ptr);
  state_->queue.enqueueUnmapMemObject(MDATA(y), mapped_ptr);
}

void OpenCL::random_uniform_impl(float lower, float upper, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  float *mapped_ptr = static_cast<float *>(
      state_->queue.enqueueMapBuffer(
        MDATA(y), CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * size, 0));
  state_->randomizer_.fill_uniform(lower, upper, size, mapped_ptr);
  state_->queue.enqueueUnmapMemObject(MDATA(y), mapped_ptr);
}

void OpenCL::random_normal_impl(float mean, float sd, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  float *mapped_ptr = static_cast<float *>(
      state_->queue.enqueueMapBuffer(
        MDATA(y), CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * size, 0));
  state_->randomizer_.fill_normal(mean, sd, size, mapped_ptr);
  state_->queue.enqueueUnmapMemObject(MDATA(y), mapped_ptr);
}

void OpenCL::random_log_normal_impl(float mean, float sd, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  float *mapped_ptr = static_cast<float *>(
      state_->queue.enqueueMapBuffer(
        MDATA(y), CL_TRUE, CL_MAP_WRITE, 0, sizeof(float) * size, 0));
  state_->randomizer_.fill_log_normal(mean, sd, size, mapped_ptr);
  state_->queue.enqueueUnmapMemObject(MDATA(y), mapped_ptr);
}

void OpenCL::pick_fw_impl(
    const Tensor &x, const std::vector<std::uint32_t> &ids,
    std::uint32_t dim, Tensor &y) {
  const std::uint32_t wy = y.shape().lower_volume(dim);
  const std::uint32_t wx = wy * x.shape()[dim];
  const std::uint32_t sx = x.shape().has_batch() * x.shape().volume();
  const std::uint32_t si = ids.size() > 1;
  const std::uint32_t sy = y.shape().volume();
  const std::uint32_t g1 = ::calc_num_blocks(sy, state_->pick_fw_group_size);
  const std::uint32_t bs = y.shape().batch();
  std::shared_ptr<void> ids_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * ids.size());
  ::write_buffer(state_->queue, ::get_buffer(ids_buf), ids.data(), ids.size());
  state_->pick_fw_kernel.setArg(0, CDATA(x));
  state_->pick_fw_kernel.setArg(1, ::get_buffer(ids_buf));
  state_->pick_fw_kernel.setArg(2, wx);
  state_->pick_fw_kernel.setArg(3, wy);
  state_->pick_fw_kernel.setArg(4, sx);
  state_->pick_fw_kernel.setArg(5, si);
  state_->pick_fw_kernel.setArg(6, sy);
  state_->pick_fw_kernel.setArg(7, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->pick_fw_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->pick_fw_group_size, bs),
      cl::NDRange(state_->pick_fw_group_size, 1));
}

void OpenCL::slice_fw_impl(
    const Tensor &x, std::uint32_t dim, std::uint32_t offset, Tensor &y) {
  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t shift = base * offset;
  const std::uint32_t span = base * y.shape()[dim];
  const std::uint32_t skip = base * x.shape()[dim];
  const std::uint32_t size = y.shape().size();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->slice_fw_group_size);
  state_->slice_fw_kernel.setArg(0, CDATA(x));
  state_->slice_fw_kernel.setArg(1, shift);
  state_->slice_fw_kernel.setArg(2, span);
  state_->slice_fw_kernel.setArg(3, skip);
  state_->slice_fw_kernel.setArg(4, size);
  state_->slice_fw_kernel.setArg(5, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->slice_fw_kernel, cl::NullRange,
      cl::NDRange(num_blocks * state_->slice_fw_group_size),
      cl::NDRange(state_->slice_fw_group_size));
}

void OpenCL::concat_fw_impl(
    const std::vector<const Tensor *> &xs, std::uint32_t dim, Tensor &y) {
  const std::uint32_t new_bs = y.shape().batch();
  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t skip = base * y.shape()[dim];
  const std::uint32_t repeat = y.shape().volume() / skip;
  std::uint32_t offset = 0;
  for (const Tensor *x : xs) {
    const std::uint32_t span = base * x->shape()[dim];
    const std::uint32_t x_size = span * repeat * x->shape().batch();
    const std::uint32_t y_size = span * repeat * new_bs;
    const std::uint32_t num_blocks = ::calc_num_blocks(
        y_size, state_->concat_fw_group_size);
    state_->concat_fw_kernel.setArg(0, CDATA(*x));
    state_->concat_fw_kernel.setArg(1, span);
    state_->concat_fw_kernel.setArg(2, skip);
    state_->concat_fw_kernel.setArg(3, x_size);
    state_->concat_fw_kernel.setArg(4, y_size);
    state_->concat_fw_kernel.setArg(5, MDATA(y));
    state_->concat_fw_kernel.setArg(6, offset);
    state_->queue.enqueueNDRangeKernel(
        state_->concat_fw_kernel, cl::NullRange,
        cl::NDRange(num_blocks * state_->concat_fw_group_size),
        cl::NDRange(state_->concat_fw_group_size), nullptr, nullptr);
    offset += span;
  }
}

void OpenCL::pick_bw_impl(
    const Tensor &gy, const std::vector<std::uint32_t> &ids,
    std::uint32_t dim, Tensor &gx) {
  const std::uint32_t wy = gy.shape().lower_volume(dim);
  const std::uint32_t wx = wy * gx.shape()[dim];
  const std::uint32_t sx = gx.shape().has_batch() * gx.shape().volume();
  const std::uint32_t si = ids.size() > 1;
  const std::uint32_t sy = gy.shape().volume();
  const std::uint32_t g1 = ::calc_num_blocks(sy, state_->concat_fw_group_size);
  const std::uint32_t bs = gy.shape().batch();
  std::shared_ptr<void> ids_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * ids.size());
  ::write_buffer(state_->queue, ::get_buffer(ids_buf), ids.data(), ids.size());
  state_->pick_bw_kernel.setArg(0, CDATA(gy));
  state_->pick_bw_kernel.setArg(1, ::get_buffer(ids_buf));
  state_->pick_bw_kernel.setArg(2, wx);
  state_->pick_bw_kernel.setArg(3, wy);
  state_->pick_bw_kernel.setArg(4, sx);
  state_->pick_bw_kernel.setArg(5, si);
  state_->pick_bw_kernel.setArg(6, sy);
  state_->pick_bw_kernel.setArg(7, MDATA(gx));
  state_->queue.enqueueNDRangeKernel(
      state_->pick_bw_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->concat_fw_group_size, bs),
      cl::NDRange(state_->concat_fw_group_size, 1));
}

void OpenCL::slice_bw_impl(
    const Tensor &gy, std::uint32_t dim, std::uint32_t offset, Tensor &gx) {
  const Shape &sx = gx.shape();
  const Shape &sy = gy.shape();
  const std::uint32_t base = sx.lower_volume(dim);
  const std::uint32_t ox = base * offset;
  const std::uint32_t wx = base * sx[dim];
  const std::uint32_t wy = base * sy[dim];
  const std::uint32_t repeat = sx.volume() / wx;
  const std::uint32_t nx = repeat * sx.batch();
  const std::uint32_t ny = repeat * sy.batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      wy * std::max(nx, ny), state_->slice_bw_group_size);
  state_->slice_bw_kernel.setArg(0, CDATA(gy));
  state_->slice_bw_kernel.setArg(1, wx);
  state_->slice_bw_kernel.setArg(2, wy);
  state_->slice_bw_kernel.setArg(3, nx);
  state_->slice_bw_kernel.setArg(4, ny);
  state_->slice_bw_kernel.setArg(5, MDATA(gx));
  state_->slice_bw_kernel.setArg(6, ox);
  state_->queue.enqueueNDRangeKernel(
      state_->slice_bw_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->slice_bw_group_size),
      cl::NDRange(state_->slice_bw_group_size));
}

#define OPENCLDEV_FW_X(name) \
void OpenCL::name##_fw_impl(const Tensor &x, Tensor &y) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = ::calc_num_blocks( \
      size, state_->name##_fw_group_size); \
  state_->name##_fw_kernel.setArg(0, CDATA(x)); \
  state_->name##_fw_kernel.setArg(1, size); \
  state_->name##_fw_kernel.setArg(2, MDATA(y)); \
  state_->queue.enqueueNDRangeKernel( \
      state_->name##_fw_kernel, cl::NullRange, \
      cl::NDRange(num_blocks * state_->name##_fw_group_size), \
      cl::NDRange(state_->name##_fw_group_size)); \
}

#define OPENCLDEV_BW_X(name) \
void OpenCL::name##_bw_impl( \
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = ::calc_num_blocks( \
      size, state_->name##_bw_group_size); \
  state_->name##_bw_kernel.setArg(0, CDATA(x)); \
  state_->name##_bw_kernel.setArg(1, CDATA(y)); \
  state_->name##_bw_kernel.setArg(2, CDATA(gy)); \
  state_->name##_bw_kernel.setArg(3, size); \
  state_->name##_bw_kernel.setArg(4, MDATA(gx)); \
  state_->queue.enqueueNDRangeKernel( \
      state_->name##_bw_kernel, cl::NullRange, \
      cl::NDRange(num_blocks * state_->name##_bw_group_size), \
      cl::NDRange(state_->name##_bw_group_size)); \
}

#define OPENCLDEV_FW_X_CONST(name) \
void OpenCL::name##_fw_impl(const Tensor &x, float k, Tensor &y) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = ::calc_num_blocks( \
      size, state_->name##_fw_group_size); \
  state_->name##_fw_kernel.setArg(0, CDATA(x)); \
  state_->name##_fw_kernel.setArg(1, k); \
  state_->name##_fw_kernel.setArg(2, size); \
  state_->name##_fw_kernel.setArg(3, MDATA(y)); \
  state_->queue.enqueueNDRangeKernel( \
      state_->name##_fw_kernel, cl::NullRange, \
      cl::NDRange(num_blocks * state_->name##_fw_group_size), \
      cl::NDRange(state_->name##_fw_group_size)); \
}

#define OPENCLDEV_BW_X_CONST(name) \
void OpenCL::name##_bw_impl( \
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = ::calc_num_blocks( \
      size, state_->name##_bw_group_size); \
  state_->name##_bw_kernel.setArg(0, CDATA(x)); \
  state_->name##_bw_kernel.setArg(1, CDATA(y)); \
  state_->name##_bw_kernel.setArg(2, CDATA(gy)); \
  state_->name##_bw_kernel.setArg(3, k); \
  state_->name##_bw_kernel.setArg(4, size); \
  state_->name##_bw_kernel.setArg(5, MDATA(gx)); \
  state_->queue.enqueueNDRangeKernel( \
      state_->name##_bw_kernel, cl::NullRange, \
      cl::NDRange(num_blocks * state_->name##_bw_group_size), \
      cl::NDRange(state_->name##_bw_group_size)); \
}

#define OPENCLDEV_FW_X_SCALAR(name) \
void OpenCL::name##_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = ::calc_num_blocks( \
      size, state_->name##_fw_group_size); \
  const std::uint32_t g2 = y.shape().batch(); \
  const std::uint32_t mbx = x.shape().has_batch(); \
  const std::uint32_t mbk = k.shape().has_batch(); \
  state_->name##_fw_kernel.setArg(0, CDATA(x)); \
  state_->name##_fw_kernel.setArg(1, CDATA(k)); \
  state_->name##_fw_kernel.setArg(2, size); \
  state_->name##_fw_kernel.setArg(3, mbx); \
  state_->name##_fw_kernel.setArg(4, mbk); \
  state_->name##_fw_kernel.setArg(5, MDATA(y)); \
  state_->queue.enqueueNDRangeKernel( \
      state_->name##_fw_kernel, cl::NullRange, \
      cl::NDRange(g1 * state_->name##_fw_group_size, g2, 1), \
      cl::NDRange(state_->name##_fw_group_size, 1, 1)); \
}

#define OPENCLDEV_FW_AB(name) \
void OpenCL::name##_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = ::calc_num_blocks( \
      size, state_->name##_fw_group_size); \
  const std::uint32_t g2 = y.shape().batch(); \
  const std::uint32_t mba = a.shape().has_batch(); \
  const std::uint32_t mbb = b.shape().has_batch(); \
  state_->name##_fw_kernel.setArg(0, CDATA(a)); \
  state_->name##_fw_kernel.setArg(1, CDATA(b)); \
  state_->name##_fw_kernel.setArg(2, size); \
  state_->name##_fw_kernel.setArg(3, mba); \
  state_->name##_fw_kernel.setArg(4, mbb); \
  state_->name##_fw_kernel.setArg(5, MDATA(y)); \
  state_->queue.enqueueNDRangeKernel( \
      state_->name##_fw_kernel, cl::NullRange, \
      cl::NDRange(g1 * state_->name##_fw_group_size, g2, 1), \
      cl::NDRange(state_->name##_fw_group_size, 1, 1)); \
}

#define OPENCLDEV_BW_AB(name) \
void OpenCL::name##_bw_impl( \
    const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, \
    Tensor &ga, Tensor &gb) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = ::calc_num_blocks( \
      size, state_->name##_bw_group_size); \
  const std::uint32_t g2 = y.shape().batch(); \
  const std::uint32_t mba = a.shape().has_batch(); \
  const std::uint32_t mbb = b.shape().has_batch(); \
  state_->name##_bw_kernel.setArg(0, CDATA(a)); \
  state_->name##_bw_kernel.setArg(1, CDATA(b)); \
  state_->name##_bw_kernel.setArg(2, CDATA(y)); \
  state_->name##_bw_kernel.setArg(3, CDATA(gy)); \
  state_->name##_bw_kernel.setArg(4, size); \
  state_->name##_bw_kernel.setArg(5, mba); \
  state_->name##_bw_kernel.setArg(6, mbb); \
  state_->name##_bw_kernel.setArg(7, MDATA(ga)); \
  state_->name##_bw_kernel.setArg(8, MDATA(gb)); \
  state_->queue.enqueueNDRangeKernel( \
      state_->name##_bw_kernel, cl::NullRange, \
      cl::NDRange(g1 * state_->name##_bw_group_size, g2, 1), \
      cl::NDRange(state_->name##_bw_group_size, 1, 1)); \
}

OPENCLDEV_FW_X(negate);
OPENCLDEV_FW_X(abs);
OPENCLDEV_FW_X(sqrt);
OPENCLDEV_FW_X(exp);
OPENCLDEV_FW_X(log);
OPENCLDEV_FW_X(tanh);
OPENCLDEV_FW_X(sigmoid);
OPENCLDEV_FW_X(softplus);
OPENCLDEV_FW_X(sin);
OPENCLDEV_FW_X(cos);
OPENCLDEV_FW_X(tan);

OPENCLDEV_BW_X(abs);
OPENCLDEV_BW_X(sqrt);
OPENCLDEV_BW_X(exp);
OPENCLDEV_BW_X(log);
OPENCLDEV_BW_X(tanh);
OPENCLDEV_BW_X(sigmoid);
OPENCLDEV_BW_X(softplus);
OPENCLDEV_BW_X(sin);
OPENCLDEV_BW_X(cos);
OPENCLDEV_BW_X(tan);

OPENCLDEV_FW_X_CONST(add_const);
OPENCLDEV_FW_X_CONST(subtract_const_r);
OPENCLDEV_FW_X_CONST(subtract_const_l);
OPENCLDEV_FW_X_CONST(multiply_const);
OPENCLDEV_FW_X_CONST(divide_const_r);
OPENCLDEV_FW_X_CONST(divide_const_l);
OPENCLDEV_FW_X_CONST(pow_const_r);
OPENCLDEV_FW_X_CONST(pow_const_l);
OPENCLDEV_FW_X_CONST(prelu);
OPENCLDEV_FW_X_CONST(elu);

void OpenCL::pown_fw_impl(const Tensor &x, std::int32_t k, Tensor &y) {
  const std::uint32_t size = x.shape().size();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->pown_fw_group_size);
  state_->pown_fw_kernel.setArg(0, CDATA(x));
  state_->pown_fw_kernel.setArg(1, k);
  state_->pown_fw_kernel.setArg(2, size);
  state_->pown_fw_kernel.setArg(3, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->pown_fw_kernel, cl::NullRange,
      cl::NDRange(num_blocks * state_->pown_fw_group_size),
      cl::NDRange(state_->pown_fw_group_size));
}

OPENCLDEV_BW_X_CONST(add_const);
OPENCLDEV_BW_X_CONST(subtract_const_r);
OPENCLDEV_BW_X_CONST(subtract_const_l);
OPENCLDEV_BW_X_CONST(multiply_const);
OPENCLDEV_BW_X_CONST(divide_const_r);
OPENCLDEV_BW_X_CONST(divide_const_l);
OPENCLDEV_BW_X_CONST(pow_const_r);
OPENCLDEV_BW_X_CONST(pow_const_l);
OPENCLDEV_BW_X_CONST(prelu);
OPENCLDEV_BW_X_CONST(elu);

void OpenCL::pown_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, std::int32_t k,
    Tensor &gx) {
  const std::uint32_t size = x.shape().size();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->pown_bw_group_size);
  state_->pown_bw_kernel.setArg(0, CDATA(x));
  state_->pown_bw_kernel.setArg(1, CDATA(y));
  state_->pown_bw_kernel.setArg(2, CDATA(gy));
  state_->pown_bw_kernel.setArg(3, k);
  state_->pown_bw_kernel.setArg(4, size);
  state_->pown_bw_kernel.setArg(5, MDATA(gx));
  state_->queue.enqueueNDRangeKernel(
      state_->pown_bw_kernel, cl::NullRange,
      cl::NDRange(num_blocks * state_->pown_bw_group_size),
      cl::NDRange(state_->pown_bw_group_size));
}

OPENCLDEV_FW_X_SCALAR(add_scalar);
OPENCLDEV_FW_X_SCALAR(subtract_scalar_r);
OPENCLDEV_FW_X_SCALAR(subtract_scalar_l);
OPENCLDEV_FW_X_SCALAR(multiply_scalar);
OPENCLDEV_FW_X_SCALAR(divide_scalar_r);
OPENCLDEV_FW_X_SCALAR(divide_scalar_l);
OPENCLDEV_FW_X_SCALAR(pow_scalar_r);
OPENCLDEV_FW_X_SCALAR(pow_scalar_l);

OPENCLDEV_FW_AB(add);
OPENCLDEV_FW_AB(subtract);
OPENCLDEV_FW_AB(multiply);
OPENCLDEV_FW_AB(divide);
OPENCLDEV_FW_AB(pow);

OPENCLDEV_BW_AB(add);
OPENCLDEV_BW_AB(subtract);
OPENCLDEV_BW_AB(multiply);
OPENCLDEV_BW_AB(divide);
OPENCLDEV_BW_AB(pow);

#undef OPENCLDEV_FW_X
#undef OPENCLDEV_BW_X
#undef OPENCLDEV_FW_X_CONST
#undef OPENCLDEV_BW_X_CONST
#undef OPENCLDEV_FW_X_SCALAR
#undef OPENCLDEV_FW_AB
#undef OPENCLDEV_BW_AB

void OpenCL::transpose_fw_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t rows = x.shape()[0];
  const std::uint32_t cols = x.shape()[1];
  const std::uint32_t bs = x.shape().batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      rows, state_->transpose_fw_group_size_x);
  const std::uint32_t g2 = ::calc_num_blocks(
      cols, state_->transpose_fw_group_size_y);
  state_->transpose_fw_kernel.setArg(0, CDATA(x));
  state_->transpose_fw_kernel.setArg(1, rows);
  state_->transpose_fw_kernel.setArg(2, cols);
  state_->transpose_fw_kernel.setArg(3, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->transpose_fw_kernel, cl::NullRange,
      cl::NDRange(
        g1 * state_->transpose_fw_group_size_x,
        g2 * state_->transpose_fw_group_size_y, bs),
      cl::NDRange(
        state_->transpose_fw_group_size_x,
        state_->transpose_fw_group_size_y, 1));
}

void OpenCL::flip_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const Shape &s = x.shape();
  const std::uint32_t n = s[dim];
  const std::uint32_t skip = s.lower_volume(dim);
  const std::uint32_t r = s.size() / n;
  const std::uint32_t g1 = ::calc_num_blocks(
      n, state_->flip_fw_group_size_x);
  const std::uint32_t g2 = ::calc_num_blocks(
      r, state_->flip_fw_group_size_y);
  state_->flip_fw_kernel.setArg(0, CDATA(x));
  state_->flip_fw_kernel.setArg(1, skip);
  state_->flip_fw_kernel.setArg(2, n);
  state_->flip_fw_kernel.setArg(3, r);
  state_->flip_fw_kernel.setArg(4, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->flip_fw_kernel, cl::NullRange,
      cl::NDRange(
          g1 * state_->flip_fw_group_size_x,
          g2 * state_->flip_fw_group_size_y),
      cl::NDRange(
          state_->flip_fw_group_size_x,
          state_->flip_fw_group_size_y));
}

// TODO(vbkaisetsu):
// Implove implementation of permute_dims.
// This function uses for-loops in the kernel code. It becomes slower than
// no-loop implementation.
void OpenCL::permute_dims_fw_impl(
    const Tensor &x, const std::vector<std::uint32_t> &perm,
    Tensor &y) {
  const std::uint32_t ndims = perm.size();
  const std::uint32_t bs = x.shape().batch();
  const std::uint32_t size = x.shape().volume();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->permute_dims_fw_group_size);
  std::vector<std::uint32_t> x_strides(ndims);
  std::vector<std::uint32_t> y_strides(ndims);
  std::uint32_t x_stride_tmp = 1;
  std::uint32_t y_stride_tmp = 1;
  for (std::uint32_t i = 0; i < ndims; ++i) {
    x_strides[ndims - i - 1] = x_stride_tmp;
    y_strides[ndims - perm[i] - 1] = y_stride_tmp;
    x_stride_tmp *= x.shape()[i];
    y_stride_tmp *= y.shape()[i];
  }
  std::shared_ptr<void> x_strides_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * x_strides.size());
  std::shared_ptr<void> y_strides_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * y_strides.size());
  if (perm.size() != 0) {
    ::write_buffer(state_->queue, ::get_buffer(x_strides_buf), x_strides.data(), x_strides.size());
    ::write_buffer(state_->queue, ::get_buffer(y_strides_buf), y_strides.data(), y_strides.size());
  }
  state_->permute_dims_fw_kernel.setArg(0, CDATA(x));
  state_->permute_dims_fw_kernel.setArg(1, ndims);
  state_->permute_dims_fw_kernel.setArg(2, ::get_buffer(x_strides_buf));
  state_->permute_dims_fw_kernel.setArg(3, ::get_buffer(y_strides_buf));
  state_->permute_dims_fw_kernel.setArg(4, size);
  state_->permute_dims_fw_kernel.setArg(5, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->permute_dims_fw_kernel, cl::NullRange,
      cl::NDRange(num_blocks * state_->transpose_fw_group_size, bs),
      cl::NDRange(state_->permute_dims_fw_group_size, 1));
}

void OpenCL::matmul_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1];
  const std::uint32_t dk = b.shape()[1];
  if (a.shape().has_batch()) {
    // Do gemm multiple times.
    const std::uint32_t a_skip = di * dj;
    const std::uint32_t b_skip = b.shape().has_batch() * dj * dk;
    const std::uint32_t y_skip = di * dk;
    const std::uint32_t bs = a.shape().batch();
    const std::vector<float> alphas(bs, 1.);
    const std::vector<float> betas(bs, 0.);
    std::vector<std::size_t> a_offsets(bs);
    std::vector<std::size_t> b_offsets(bs);
    std::vector<std::size_t> y_offsets(bs);
    for (std::uint32_t n = 0; n < bs; ++n) {
      a_offsets[n] = n * a_skip;
      b_offsets[n] = n * b_skip;
      y_offsets[n] = n * y_skip;
    }
    clblast::GemmBatched(
      clblast::Layout::kColMajor,
      clblast::Transpose::kNo, clblast::Transpose::kNo,
      di, dk, dj,
      alphas.data(),
      CDATA(a)(), a_offsets.data(), di,
      CDATA(b)(), b_offsets.data(), dj,
      betas.data(),
      MDATA(y)(), y_offsets.data(), di,
      bs,
      &state_->queue(), nullptr);
  } else {
    // Do gemm only once to calculate the product with a combined matrix.
    const float alpha = 1.;
    const float beta = 0.;
    clblast::Gemm(
      clblast::Layout::kColMajor,
      clblast::Transpose::kNo, clblast::Transpose::kNo,
      di, dk * b.shape().batch(), dj,
      alpha,
      CDATA(a)(), 0, di,
      CDATA(b)(), 0, dj,
      beta,
      MDATA(y)(), 0, di,
      &state_->queue(), nullptr);
  }
}

void OpenCL::transpose_bw_impl(
    const Tensor &, const Tensor &, const Tensor &gy, Tensor &gx) {
  const std::uint32_t rows = gx.shape()[0];
  const std::uint32_t cols = gx.shape()[1];
  const std::uint32_t bs = gx.shape().batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      rows, state_->transpose_bw_group_size_x);
  const std::uint32_t g2 = ::calc_num_blocks(
      cols, state_->transpose_bw_group_size_y);
  state_->transpose_bw_kernel.setArg(0, CDATA(gy));
  state_->transpose_bw_kernel.setArg(1, rows);
  state_->transpose_bw_kernel.setArg(2, cols);
  state_->transpose_bw_kernel.setArg(3, MDATA(gx));
  state_->queue.enqueueNDRangeKernel(
      state_->transpose_bw_kernel, cl::NullRange,
      cl::NDRange(
        g1 * state_->transpose_bw_group_size_x,
        g2 * state_->transpose_bw_group_size_y, bs),
      cl::NDRange(
        state_->transpose_bw_group_size_x,
        state_->transpose_bw_group_size_y, 1));
}

void OpenCL::flip_bw_impl(const Tensor &gy, std::uint32_t dim, Tensor &gx) {
  const Shape &s = gy.shape();
  const std::uint32_t n = s[dim];
  const std::uint32_t skip = s.lower_volume(dim);
  const std::uint32_t r = s.size() / n;
  const std::uint32_t g1 = ::calc_num_blocks(
      n, state_->flip_bw_group_size_x);
  const std::uint32_t g2 = ::calc_num_blocks(
      r, state_->flip_bw_group_size_y);
  state_->flip_bw_kernel.setArg(0, CDATA(gy));
  state_->flip_bw_kernel.setArg(1, skip);
  state_->flip_bw_kernel.setArg(2, n);
  state_->flip_bw_kernel.setArg(3, r);
  state_->flip_bw_kernel.setArg(4, MDATA(gx));
  state_->queue.enqueueNDRangeKernel(
      state_->flip_bw_kernel, cl::NullRange,
      cl::NDRange(
          g1 * state_->flip_bw_group_size_x,
          g2 * state_->flip_bw_group_size_y),
      cl::NDRange(
          state_->flip_bw_group_size_x,
          state_->flip_bw_group_size_y));
}

void OpenCL::permute_dims_bw_impl(
    const Tensor &, const Tensor &, const Tensor &gy,
    const std::vector<std::uint32_t> &perm, Tensor &gx) {
  const std::uint32_t ndims = perm.size();
  const std::uint32_t bs = gx.shape().batch();
  const std::uint32_t size = gx.shape().volume();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->permute_dims_bw_group_size);
  std::vector<std::uint32_t> x_strides(ndims);
  std::vector<std::uint32_t> y_strides(ndims);
  std::uint32_t x_stride_tmp = 1;
  std::uint32_t y_stride_tmp = 1;
  for (std::uint32_t i = 0; i < ndims; ++i) {
    x_strides[ndims - i - 1] = x_stride_tmp;
    y_strides[ndims - perm[i] - 1] = y_stride_tmp;
    x_stride_tmp *= gx.shape()[i];
    y_stride_tmp *= gy.shape()[i];
  }
  std::shared_ptr<void> x_strides_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * x_strides.size());
  std::shared_ptr<void> y_strides_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * y_strides.size());
  if (perm.size() != 0) {
    ::write_buffer(state_->queue, ::get_buffer(x_strides_buf), x_strides.data(), x_strides.size());
    ::write_buffer(state_->queue, ::get_buffer(y_strides_buf), y_strides.data(), y_strides.size());
  }
  state_->permute_dims_bw_kernel.setArg(0, CDATA(gy));
  state_->permute_dims_bw_kernel.setArg(1, ndims);
  state_->permute_dims_bw_kernel.setArg(2, ::get_buffer(x_strides_buf));
  state_->permute_dims_bw_kernel.setArg(3, ::get_buffer(y_strides_buf));
  state_->permute_dims_bw_kernel.setArg(4, size);
  state_->permute_dims_bw_kernel.setArg(5, MDATA(gx));
  state_->queue.enqueueNDRangeKernel(
      state_->permute_dims_bw_kernel, cl::NullRange,
      cl::NDRange(num_blocks * state_->transpose_bw_group_size, bs),
      cl::NDRange(state_->permute_dims_bw_group_size, 1));
}

void OpenCL::matmul_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  // ga += gy . b^T
  // gb += a^T . gy
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1];
  const std::uint32_t dk = b.shape()[1];
  if (a.shape().has_batch()) {
    // Do gemm multiple times.
    const std::uint32_t a_skip = di * dj;
    const std::uint32_t b_skip = b.shape().has_batch() * dj * dk;
    const std::uint32_t y_skip = di * dk;
    const std::uint32_t bs = a.shape().batch();
    const std::vector<float> alphas(bs, 1.);
    const std::vector<float> betas(bs, 1.);
    std::vector<std::size_t> a_offsets(bs);
    std::vector<std::size_t> b_offsets(bs);
    std::vector<std::size_t> y_offsets(bs);
    for (std::uint32_t n = 0; n < bs; ++n) {
      a_offsets[n] = n * a_skip;
      b_offsets[n] = n * b_skip;
      y_offsets[n] = n * y_skip;
    }
    clblast::GemmBatched(
      clblast::Layout::kColMajor,
      clblast::Transpose::kNo, clblast::Transpose::kYes,
      di, dj, dk,
      alphas.data(),
      CDATA(gy)(), y_offsets.data(), di,
      CDATA(b)(), b_offsets.data(), dj,
      betas.data(),
      MDATA(ga)(), a_offsets.data(), di,
      bs,
      &state_->queue(), nullptr);
    if (b_skip > 0) {
      clblast::GemmBatched(
        clblast::Layout::kColMajor,
        clblast::Transpose::kYes, clblast::Transpose::kNo,
        dj, dk, di,
        alphas.data(),
        CDATA(a)(), a_offsets.data(), di,
        CDATA(gy)(), y_offsets.data(), di,
        betas.data(),
        MDATA(gb)(), b_offsets.data(), dj,
        bs,
        &state_->queue(), nullptr);
    } else {
      // NOTE(vbkaisetsu):
      // `clblast::GemmBatched` can not be used due to a data race against
      // shared values in `b` by multiple GEMM operations.
      const float alpha = 1.;
      const float beta = 1.;
      for (std::uint32_t n = 0; n < bs; ++n) {
        clblast::Gemm(
          clblast::Layout::kColMajor,
          clblast::Transpose::kYes, clblast::Transpose::kNo,
          dj, dk, di,
          alpha,
          CDATA(a)(), n * a_skip, di,
          CDATA(gy)(), n * y_skip, di,
          beta,
          MDATA(gb)(), n * b_skip, dj,
          &state_->queue(), nullptr);
      }
    }
  } else {
    // Do gemm only once to calculate the product with a combined matrix.
    const float alpha = 1.;
    const float beta = 1.;
    clblast::Gemm(
      clblast::Layout::kColMajor,
      clblast::Transpose::kNo, clblast::Transpose::kYes,
      di, dj, dk * b.shape().batch(),
      alpha,
      CDATA(gy)(), 0, di,
      CDATA(b)(), 0, dj,
      beta,
      MDATA(ga)(), 0, di,
      &state_->queue(), nullptr);
    clblast::Gemm(
      clblast::Layout::kColMajor,
      clblast::Transpose::kYes, clblast::Transpose::kNo,
      dj, dk * b.shape().batch(), di,
      alpha,
      CDATA(a)(), 0, di,
      CDATA(gy)(), 0, di,
      beta,
      MDATA(gb)(), 0, dj,
      &state_->queue(), nullptr);
  }
}

void OpenCL::max_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t group_size = std::min(state_->max_fw_group_size, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      state_->max_fw_kernel[m].setArg(0, CDATA(x)); \
      state_->max_fw_kernel[m].setArg(1, s); \
      state_->max_fw_kernel[m].setArg(2, n); \
      state_->max_fw_kernel[m].setArg(3, MDATA(y)); \
      state_->queue.enqueueNDRangeKernel( \
          state_->max_fw_kernel[m], \
          cl::NullRange, cl::NDRange(r * k), cl::NDRange(k)); \
      break;
    CASE(1024, 10);
    CASE(512, 9);
    CASE(256, 8);
    CASE(128, 7);
    CASE(64, 6);
    CASE(32, 5);
    CASE(16, 4);
    CASE(8, 3);
    CASE(4, 2);
    CASE(2, 1);
    CASE(1, 0);
#undef CASE
  }
}

void OpenCL::max_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy,
    std::uint32_t dim, Tensor &gx) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t group_size = std::min(state_->max_bw_group_size, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      state_->max_bw_kernel[m].setArg(0, CDATA(x)); \
      state_->max_bw_kernel[m].setArg(1, CDATA(y)); \
      state_->max_bw_kernel[m].setArg(2, CDATA(gy)); \
      state_->max_bw_kernel[m].setArg(3, s); \
      state_->max_bw_kernel[m].setArg(4, n); \
      state_->max_bw_kernel[m].setArg(5, MDATA(gx)); \
      state_->queue.enqueueNDRangeKernel( \
          state_->max_bw_kernel[m], \
          cl::NullRange, cl::NDRange(r * k), cl::NDRange(k)); \
      break;
    CASE(1024, 10);
    CASE(512, 9);
    CASE(256, 8);
    CASE(128, 7);
    CASE(64, 6);
    CASE(32, 5);
    CASE(16, 4);
    CASE(8, 3);
    CASE(4, 2);
    CASE(2, 1);
    CASE(1, 0);
#undef CASE
  }
}

void OpenCL::min_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t group_size = std::min(state_->min_fw_group_size, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      state_->min_fw_kernel[m].setArg(0, CDATA(x)); \
      state_->min_fw_kernel[m].setArg(1, s); \
      state_->min_fw_kernel[m].setArg(2, n); \
      state_->min_fw_kernel[m].setArg(3, MDATA(y)); \
      state_->queue.enqueueNDRangeKernel( \
          state_->min_fw_kernel[m], \
          cl::NullRange, cl::NDRange(r * k), cl::NDRange(k)); \
      break;
    CASE(1024, 10);
    CASE(512, 9);
    CASE(256, 8);
    CASE(128, 7);
    CASE(64, 6);
    CASE(32, 5);
    CASE(16, 4);
    CASE(8, 3);
    CASE(4, 2);
    CASE(2, 1);
    CASE(1, 0);
#undef CASE
  }
}

void OpenCL::min_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy,
    std::uint32_t dim, Tensor &gx) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t group_size = std::min(state_->min_bw_group_size, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      state_->min_bw_kernel[m].setArg(0, CDATA(x)); \
      state_->min_bw_kernel[m].setArg(1, CDATA(y)); \
      state_->min_bw_kernel[m].setArg(2, CDATA(gy)); \
      state_->min_bw_kernel[m].setArg(3, s); \
      state_->min_bw_kernel[m].setArg(4, n); \
      state_->min_bw_kernel[m].setArg(5, MDATA(gx)); \
      state_->queue.enqueueNDRangeKernel( \
          state_->min_bw_kernel[m], \
          cl::NullRange, cl::NDRange(r * k), cl::NDRange(k)); \
      break;
    CASE(1024, 10);
    CASE(512, 9);
    CASE(256, 8);
    CASE(128, 7);
    CASE(64, 6);
    CASE(32, 5);
    CASE(16, 4);
    CASE(8, 3);
    CASE(4, 2);
    CASE(2, 1);
    CASE(1, 0);
#undef CASE
  }
}

void OpenCL::sum_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t group_size = std::min(state_->sum_fw_group_size, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      state_->sum_fw_kernel[m].setArg(0, CDATA(x)); \
      state_->sum_fw_kernel[m].setArg(1, s); \
      state_->sum_fw_kernel[m].setArg(2, n); \
      state_->sum_fw_kernel[m].setArg(3, MDATA(y)); \
      state_->queue.enqueueNDRangeKernel( \
          state_->sum_fw_kernel[m], \
          cl::NullRange, cl::NDRange(r * k), cl::NDRange(k)); \
      break;
    CASE(1024, 10);
    CASE(512, 9);
    CASE(256, 8);
    CASE(128, 7);
    CASE(64, 6);
    CASE(32, 5);
    CASE(16, 4);
    CASE(8, 3);
    CASE(4, 2);
    CASE(2, 1);
    CASE(1, 0);
#undef CASE
  }
}

void OpenCL::logsumexp_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t group_size = std::min(state_->logsumexp_fw_group_size, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      state_->logsumexp_fw_kernel[m].setArg(0, CDATA(x)); \
      state_->logsumexp_fw_kernel[m].setArg(1, s); \
      state_->logsumexp_fw_kernel[m].setArg(2, n); \
      state_->logsumexp_fw_kernel[m].setArg(3, MDATA(y)); \
      state_->queue.enqueueNDRangeKernel( \
          state_->logsumexp_fw_kernel[m], \
          cl::NullRange, cl::NDRange(r * k), cl::NDRange(k)); \
      break;
    CASE(1024, 10);
    CASE(512, 9);
    CASE(256, 8);
    CASE(128, 7);
    CASE(64, 6);
    CASE(32, 5);
    CASE(16, 4);
    CASE(8, 3);
    CASE(4, 2);
    CASE(2, 1);
    CASE(1, 0);
#undef CASE
  }
}

void OpenCL::broadcast_fw_impl(
    const Tensor &x, std::uint32_t dim, std::uint32_t size, Tensor &y) {
  const std::uint32_t skip1 = y.shape().lower_volume(dim);
  const std::uint32_t skip2 = skip1 * size;
  const std::uint32_t total = y.shape().size();
  const std::uint32_t g1 = ::calc_num_blocks(
      total, state_->broadcast_fw_group_size);
  state_->broadcast_fw_kernel.setArg(0, CDATA(x));
  state_->broadcast_fw_kernel.setArg(1, skip1);
  state_->broadcast_fw_kernel.setArg(2, skip2);
  state_->broadcast_fw_kernel.setArg(3, total);
  state_->broadcast_fw_kernel.setArg(4, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->broadcast_fw_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->broadcast_fw_group_size),
      cl::NDRange(state_->broadcast_fw_group_size));
}

void OpenCL::batch_pick_fw_impl(const Tensor &x, const std::vector<std::uint32_t> &ids, Tensor &y) {
  const std::uint32_t si = ids.size() > 1;
  const std::uint32_t sy = y.shape().volume();
  const std::uint32_t g1 = ::calc_num_blocks(sy, state_->batch_pick_fw_group_size);
  const std::uint32_t bs = y.shape().batch();
  std::shared_ptr<void> ids_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * ids.size());
  ::write_buffer(state_->queue, ::get_buffer(ids_buf), ids.data(), ids.size());
  state_->batch_pick_fw_kernel.setArg(0, CDATA(x));
  state_->batch_pick_fw_kernel.setArg(1, ::get_buffer(ids_buf));
  state_->batch_pick_fw_kernel.setArg(2, si);
  state_->batch_pick_fw_kernel.setArg(3, sy);
  state_->batch_pick_fw_kernel.setArg(4, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->batch_pick_fw_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->batch_pick_fw_group_size, bs),
      cl::NDRange(state_->batch_pick_fw_group_size, 1));
}

void OpenCL::batch_slice_fw_impl(
    const Tensor &x, std::uint32_t offset, Tensor &y) {
  const std::uint32_t volume = y.shape().volume();
  const std::uint32_t shift = volume * offset;
  const std::uint32_t size = y.shape().size();
  const std::uint32_t num_blocks = ::calc_num_blocks(
      size, state_->batch_slice_fw_group_size);
  state_->batch_slice_fw_kernel.setArg(0, CDATA(x));
  state_->batch_slice_fw_kernel.setArg(1, shift);
  state_->batch_slice_fw_kernel.setArg(2, size);
  state_->batch_slice_fw_kernel.setArg(3, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->batch_slice_fw_kernel, cl::NullRange,
      cl::NDRange(num_blocks * state_->batch_slice_fw_group_size),
      cl::NDRange(state_->batch_slice_fw_group_size));
}

void OpenCL::batch_concat_fw_impl(
    const std::vector<const Tensor *> &xs, Tensor &y) {
  std::uint32_t offset = 0;
  for (const Tensor *x : xs) {
    const std::uint32_t span = x->shape().size();
    const std::uint32_t num_blocks = ::calc_num_blocks(
        span, state_->batch_concat_fw_group_size);
    state_->batch_concat_fw_kernel.setArg(0, CDATA(*x));
    state_->batch_concat_fw_kernel.setArg(1, span);
    state_->batch_concat_fw_kernel.setArg(2, MDATA(y));
    state_->batch_concat_fw_kernel.setArg(3, offset);
    state_->queue.enqueueNDRangeKernel(
        state_->batch_concat_fw_kernel, cl::NullRange,
        cl::NDRange(num_blocks * state_->batch_concat_fw_group_size),
        cl::NDRange(state_->batch_concat_fw_group_size), nullptr, nullptr);
    offset += span;
  }
}

void OpenCL::batch_sum_fw_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  const std::uint32_t batch = x.shape().batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, state_->batch_sum_fw_group_size);
  state_->batch_sum_fw_kernel.setArg(0, CDATA(x));
  state_->batch_sum_fw_kernel.setArg(1, size);
  state_->batch_sum_fw_kernel.setArg(2, batch);
  state_->batch_sum_fw_kernel.setArg(3, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->batch_sum_fw_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->batch_sum_fw_group_size),
      cl::NDRange(state_->batch_sum_fw_group_size));
}

void OpenCL::batch_pick_bw_impl(const Tensor &gy, const std::vector<std::uint32_t> &ids, Tensor &gx) {
  const std::uint32_t si = ids.size() > 1;
  const std::uint32_t sy = gy.shape().volume();
  const std::uint32_t g1 = ::calc_num_blocks(sy, state_->batch_pick_bw_group_size);
  const std::uint32_t bs = gy.shape().batch();
  std::shared_ptr<void> ids_buf = state_->pool.allocate(
      sizeof(std::uint32_t) * ids.size());
  ::write_buffer(state_->queue, ::get_buffer(ids_buf), ids.data(), ids.size());
  state_->batch_pick_bw_kernel.setArg(0, CDATA(gy));
  state_->batch_pick_bw_kernel.setArg(1, ::get_buffer(ids_buf));
  state_->batch_pick_bw_kernel.setArg(2, si);
  state_->batch_pick_bw_kernel.setArg(3, sy);
  state_->batch_pick_bw_kernel.setArg(4, MDATA(gx));
  state_->queue.enqueueNDRangeKernel(
      state_->batch_pick_bw_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->batch_pick_bw_group_size, bs),
      cl::NDRange(state_->batch_pick_bw_group_size, 1));
}

void OpenCL::batch_slice_bw_impl(const Tensor &gy, std::uint32_t offset, Tensor &gx) {
  const std::uint32_t volume = gy.shape().volume();
  const std::uint32_t shift = volume * offset;
  const std::uint32_t size = gy.shape().size();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, state_->batch_slice_bw_group_size);
  state_->batch_slice_bw_kernel.setArg(0, CDATA(gy));
  state_->batch_slice_bw_kernel.setArg(1, size);
  state_->batch_slice_bw_kernel.setArg(2, MDATA(gx));
  state_->batch_slice_bw_kernel.setArg(3, shift);
  state_->queue.enqueueNDRangeKernel(
      state_->batch_slice_bw_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->batch_slice_bw_group_size),
      cl::NDRange(state_->batch_slice_bw_group_size));
}

void OpenCL::conv2d_fw_impl(const Tensor &, const Tensor &,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    Tensor &) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

void OpenCL::conv2d_bw_impl(
    const Tensor &, const Tensor &, const Tensor &, const Tensor &,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    Tensor &, Tensor &) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

void OpenCL::max_pool2d_fw_impl(
    const Tensor &,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    Tensor &) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

void OpenCL::max_pool2d_bw_impl(
    const Tensor &, const Tensor &, const Tensor &,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    Tensor &) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

void OpenCL::inplace_multiply_const_impl(float k, Tensor &x) {
  const std::uint32_t size = x.shape().size();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, state_->inplace_multiply_const_group_size);
  state_->inplace_multiply_const_kernel.setArg(0, k);
  state_->inplace_multiply_const_kernel.setArg(1, size);
  state_->inplace_multiply_const_kernel.setArg(2, MDATA(x));
  state_->queue.enqueueNDRangeKernel(
      state_->inplace_multiply_const_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->inplace_multiply_const_group_size),
      cl::NDRange(state_->inplace_multiply_const_group_size));
}

void OpenCL::inplace_add_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t size = y.shape().volume();
  const std::uint32_t mbx = x.shape().has_batch();
  const std::uint32_t mby = y.shape().has_batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, state_->inplace_add_group_size);
  const std::uint32_t bs = std::max(x.shape().batch(), y.shape().batch());
  state_->inplace_add_kernel.setArg(0, CDATA(x));
  state_->inplace_add_kernel.setArg(1, size);
  state_->inplace_add_kernel.setArg(2, mbx);
  state_->inplace_add_kernel.setArg(3, mby);
  state_->inplace_add_kernel.setArg(4, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->inplace_add_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->inplace_add_group_size, bs, 1),
      cl::NDRange(state_->inplace_add_group_size, 1, 1));
}

void OpenCL::inplace_subtract_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t size = y.shape().volume();
  const std::uint32_t mbx = x.shape().has_batch();
  const std::uint32_t mby = y.shape().has_batch();
  const std::uint32_t g1 = ::calc_num_blocks(
      size, state_->inplace_subtract_group_size);
  const std::uint32_t bs = std::max(x.shape().batch(), y.shape().batch());
  state_->inplace_subtract_kernel.setArg(0, CDATA(x));
  state_->inplace_subtract_kernel.setArg(1, size);
  state_->inplace_subtract_kernel.setArg(2, mbx);
  state_->inplace_subtract_kernel.setArg(3, mby);
  state_->inplace_subtract_kernel.setArg(4, MDATA(y));
  state_->queue.enqueueNDRangeKernel(
      state_->inplace_subtract_kernel, cl::NullRange,
      cl::NDRange(g1 * state_->inplace_subtract_group_size, bs, 1),
      cl::NDRange(state_->inplace_subtract_group_size, 1, 1));
}

}  // namespace devices
}  // namespace primitiv
