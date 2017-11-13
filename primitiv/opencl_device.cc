#include <config.h>

#include <CL/cl.hpp>

#include <iostream>
#include <random>
#include <primitiv/error.h>
#include <primitiv/opencl_device.h>

namespace primitiv {
namespace devices {

#define DATA(x) *((cl::Buffer *) ((x).data()))

std::string OpenCL::kernel_code_generator() {
  std::ostringstream ss;
  for(std::uint32_t group_size = 1; group_size <= 1024; group_size <<= 1) {
    ss <<
"kernel void sum_fw_kernel_" << group_size <<
"(constant float *px, constant unsigned *skip_p, constant unsigned *n_p, global float *py) {\n"
"#define GROUP_SIZE " << group_size << R"EOS(
  unsigned bid = get_group_id(0);
  unsigned tid = get_local_id(0);
  unsigned skip = skip_p[0];
  unsigned n = n_p[0];
  local float temp[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  temp[tid] = 0;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) temp[tid] += px[i * skip];
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
#define REDUCE(k) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) temp[tid] += temp[tid + k]; \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
  }
  REDUCE(512)
  REDUCE(256)
  REDUCE(128)
  REDUCE(64)
  REDUCE(32)
  REDUCE(16)
  REDUCE(8)
  REDUCE(4)
  REDUCE(2)
  REDUCE(1)
#undef REDUCE
  if (tid == 0) py[bid] = temp[0];
#undef GROUP_SIZE
}
)EOS";
  }
  for(std::uint32_t group_size = 1; group_size <= 1024; group_size <<= 1) {
    ss <<
"kernel void argmax_kernel_" << group_size <<
"(constant float *px, constant unsigned *skip_p, constant unsigned *n_p, global unsigned *py) {\n"
"#define GROUP_SIZE " << group_size << R"EOS(
  unsigned bid = get_group_id(0);
  unsigned tid = get_local_id(0);
  unsigned skip = skip_p[0];
  unsigned n = n_p[0];
  local float max_val[GROUP_SIZE];
  local unsigned argmax_val[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  max_val[tid] = -1e38;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) {
    float val = px[i * skip];
    if (val > max_val[tid]) {
      max_val[tid] = val;
      argmax_val[tid] = i;
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
#define REDUCE(k) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) { \
      if (max_val[tid + k] > max_val[tid]) { \
        max_val[tid] = max_val[tid + k]; \
        argmax_val[tid] = argmax_val[tid + k]; \
      } \
    } \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
  }
  REDUCE(512)
  REDUCE(256)
  REDUCE(128)
  REDUCE(64)
  REDUCE(32)
  REDUCE(16)
  REDUCE(8)
  REDUCE(4)
  REDUCE(2)
  REDUCE(1)
#undef REDUCE
  if (tid == 0) py[bid] = argmax_val[0];
#undef GROUP_SIZE
}
)EOS";
  }
  for(std::uint32_t group_size = 1; group_size <= 1024; group_size <<= 1) {
    ss <<
"kernel void argmin_kernel_" << group_size <<
"(constant float *px, constant unsigned *skip_p, constant unsigned *n_p, global unsigned *py) {\n"
"#define GROUP_SIZE " << group_size << R"EOS(
  unsigned bid = get_group_id(0);
  unsigned tid = get_local_id(0);
  unsigned skip = skip_p[0];
  unsigned n = n_p[0];
  local float max_val[GROUP_SIZE];
  local unsigned argmax_val[GROUP_SIZE];
  px += bid % skip + (bid / skip) * skip * n;
  max_val[tid] = 1e38;
  for (unsigned i = tid; i < n; i += GROUP_SIZE) {
    float val = px[i * skip];
    if (val < max_val[tid]) {
      max_val[tid] = val;
      argmax_val[tid] = i;
    }
  }
  work_group_barrier(CLK_LOCAL_MEM_FENCE);
#define REDUCE(k) \
  if (GROUP_SIZE >= k << 1) { \
    if (tid < k) { \
      if (max_val[tid + k] < max_val[tid]) { \
        max_val[tid] = max_val[tid + k]; \
        argmax_val[tid] = argmax_val[tid + k]; \
      } \
    } \
    work_group_barrier(CLK_LOCAL_MEM_FENCE); \
  }
  REDUCE(512)
  REDUCE(256)
  REDUCE(128)
  REDUCE(64)
  REDUCE(32)
  REDUCE(16)
  REDUCE(8)
  REDUCE(4)
  REDUCE(2)
  REDUCE(1)
#undef REDUCE
  if (tid == 0) py[bid] = argmax_val[0];
#undef GROUP_SIZE
}
)EOS";
  }
  return ss.str();
}

std::uint32_t num_platforms() {
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  return all_platforms.size();
}

std::uint32_t num_devices(std::uint32_t platform_id) {\
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if (all_platforms.size() == 0) {
    THROW_ERROR("No platforms found. Check OpenCL installation!");
  }
  cl::Platform platform = all_platforms[platform_id];\
  std::vector<cl::Device> all_devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  return all_devices.size();
}

OpenCL::OpenCL(std::uint32_t platform_id, std::uint32_t device_id) {
  std::int32_t error = CL_SUCCESS;
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if (all_platforms.size() == 0) {
    THROW_ERROR("No platforms found. Check OpenCL installation!");
  }
  cl::Platform platform = all_platforms[platform_id];
  std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

  std::vector<cl::Device> all_devices;
  platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if(all_devices.size() == 0){
    THROW_ERROR("No devices found. Check OpenCL installation!");
  }
  dev_id_ = device_id;
  device_ = all_devices[dev_id_];
  context_ = cl::Context({device_});
  std::cout << "Using device: " << device_.getInfo<CL_DEVICE_NAME>() << std::endl;

  cl::Program::Sources sources;
  std::string kernel_code = kernel_code_generator();

  sources.push_back({kernel_code.c_str(), kernel_code.length()});
  cl::Program program(context_, sources);
  if (program.build({device_}, "-cl-std=CL2.0") != CL_SUCCESS) {
    std::cerr << " Error building: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device_) << std::endl;
    THROW_ERROR("Error!");
  }
  for (std::uint32_t i = 0; i <= 10; ++i) {
    std::ostringstream ss;
    ss << "argmax_kernel_" << (1 << i);
    argmax_kernel_[i] = cl::Kernel(program, ss.str().c_str(), &error);
  }
  for (std::uint32_t i = 0; i <= 10; ++i) {
    std::ostringstream ss;
    ss << "argmin_kernel_" << (1 << i);
    argmin_kernel_[i] = cl::Kernel(program, ss.str().c_str(), &error);
  }
  for (std::uint32_t i = 0; i <= 10; ++i) {
    std::ostringstream ss;
    ss << "sum_fw_kernel_" << (1 << i);
    sum_fw_kernel_[i] = cl::Kernel(program, ss.str().c_str(), &error);
  }
}

OpenCL::~OpenCL() {
  // Nothing to do for now.
}

void OpenCL::dump_description() const {
  std::cerr << "Device " << this << ':' << std::endl;
  std::cerr << "  Type: OpenCL" << std::endl;
  // TODO
}

std::shared_ptr<void> OpenCL::new_handle(const Shape &shape) {
  cl_int error = CL_SUCCESS;
  const std::uint32_t mem_size = sizeof(float) * shape.size();
  cl::Buffer *data = new cl::Buffer(context_,
            CL_MEM_READ_WRITE,
            mem_size,
            NULL, &error);
  if (error != CL_SUCCESS) {
    THROW_ERROR("Memory allocation failed. Requested size: " << mem_size);
  }
  return std::shared_ptr<void>(data, [](cl::Buffer *buffer){delete buffer;});
}

std::vector<float> OpenCL::tensor_to_vector_impl(const Tensor &x) {
  cl_int error = CL_SUCCESS;
  const std::uint32_t num_elements = x.shape().size();
  std::vector<float> ret(num_elements);
  cl::CommandQueue queue(context_, device_, 0, &error);
  queue.enqueueReadBuffer(*((cl::Buffer *) x.data()), CL_TRUE, 0,
            sizeof(cl_float) * num_elements, &ret.at(0), NULL, NULL);
  return ret;
}

void OpenCL::reset_tensor_by_array_impl(const float values[], Tensor &x) {
  std::int32_t error = CL_SUCCESS;
  const std::uint32_t size = x.shape().size();
  cl::CommandQueue queue(context_, device_, 0, &error);
  queue.enqueueWriteBuffer(*((cl::Buffer *) x.data()), CL_TRUE, 0,
            sizeof(float) * size, values, NULL, NULL);
}

std::vector<std::uint32_t> OpenCL::argmax_impl(const Tensor &x, std::uint32_t dim) {
  const Shape &shape = x.shape();
  std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  std::uint32_t s = shape.lower_volume(dim);
  std::uint32_t group_size = argmax_kernel_[0].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  cl_int error = CL_SUCCESS;
  while (group_size >> 1 >= n) group_size >>= 1;
  cl::CommandQueue queue(context_, device_, 0, &error);
  cl::Buffer mem_s = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(cl_uint), &s, &error);
  cl::Buffer mem_n = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(cl_uint), &n, &error);
  cl::Buffer py = cl::Buffer(context_,
      CL_MEM_WRITE_ONLY,
      sizeof(cl_uint) * r, NULL, &error);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      argmax_kernel_[m].setArg(0, DATA(x)); \
      argmax_kernel_[m].setArg(1, mem_s); \
      argmax_kernel_[m].setArg(2, mem_n); \
      argmax_kernel_[m].setArg(3, py); \
      queue.enqueueNDRangeKernel(argmax_kernel_[m], cl::NullRange, cl::NDRange(r * k), cl::NDRange(k), NULL, NULL); \
      queue.finish();; break;
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
  queue.enqueueReadBuffer(py, CL_TRUE, 0,
            sizeof(cl_uint) * r, &ret.at(0), NULL, NULL);
  return ret;
}

std::vector<std::uint32_t> OpenCL::argmin_impl(const Tensor &x, std::uint32_t dim) {
  const Shape &shape = x.shape();
  std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  std::uint32_t s = shape.lower_volume(dim);
  std::uint32_t group_size = argmin_kernel_[0].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  cl_int error = CL_SUCCESS;
  while (group_size >> 1 >= n) group_size >>= 1;
  cl::CommandQueue queue(context_, device_, 0, &error);
  cl::Buffer mem_s = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(cl_uint), &s, &error);
  cl::Buffer mem_n = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(cl_uint), &n, &error);
  cl::Buffer py = cl::Buffer(context_,
      CL_MEM_WRITE_ONLY,
      sizeof(cl_uint) * r, NULL, &error);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      argmin_kernel_[m].setArg(0, DATA(x)); \
      argmin_kernel_[m].setArg(1, mem_s); \
      argmin_kernel_[m].setArg(2, mem_n); \
      argmin_kernel_[m].setArg(3, py); \
      queue.enqueueNDRangeKernel(argmin_kernel_[m], cl::NullRange, cl::NDRange(r * k), cl::NDRange(k), NULL, NULL); \
      queue.finish();; break;
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
  queue.enqueueReadBuffer(py, CL_TRUE, 0,
            sizeof(cl_uint) * r, &ret.at(0), NULL, NULL);
  return ret;
}

void OpenCL::sum_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t group_size = sum_fw_kernel_[0].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  cl_int error = CL_SUCCESS;
  while (group_size >> 1 >= n) group_size >>= 1;
  cl::CommandQueue queue(context_, device_, 0, &error);
  cl::Buffer mem_s = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(cl_uint), &s, &error);
  cl::Buffer mem_n = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      sizeof(cl_uint), &n, &error);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      sum_fw_kernel_[m].setArg(0, DATA(x)); \
      sum_fw_kernel_[m].setArg(1, mem_s); \
      sum_fw_kernel_[m].setArg(2, mem_n); \
      sum_fw_kernel_[m].setArg(3, DATA(y)); \
      queue.enqueueNDRangeKernel(sum_fw_kernel_[m], cl::NullRange, cl::NDRange(r * k), cl::NDRange(k), NULL, NULL); \
      queue.finish();; break;
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

}
}
