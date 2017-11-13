#include <config.h>

#include <CL/cl.hpp>

#include <iostream>
#include <random>
#include <primitiv/error.h>
#include <primitiv/opencl_device.h>

namespace primitiv {
namespace devices {

#define GRID_SIZE(x, threads) (((x) + (threads) - 1) / (threads))
#define CDATA(x) *(static_cast<const cl::Buffer *>((x).data()))

#define SET_ARG_HOST_SCALAR(kernel, idx, type, var) \
  cl::Buffer opencl_mem_##var = cl::Buffer(context_, \
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, \
      sizeof(type), &var, &error); \
  kernel.setArg(idx, opencl_mem_##var);

#define SET_ARG_HOST_VECTOR(kernel, idx, type, var) \
  cl::Buffer opencl_mem_##var = cl::Buffer(context_, \
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, \
      sizeof(type) * var.size(), const_cast<type*>(var.data()), &error); \
  kernel.setArg(idx, opencl_mem_##var);

std::string OpenCL::kernel_code_generator() {
  std::ostringstream ss;
  for(std::uint32_t group_size = 1; group_size <= 1024; group_size <<= 1) {
    ss <<
"kernel void sum_fw_kernel_" << group_size <<
"(constant float *px, constant unsigned *skip_p, constant unsigned *n_p, global float *py) {\n"
"#define GROUP_SIZE " << group_size << R"EOS(
  unsigned skip = skip_p[0];
  unsigned n = n_p[0];
  unsigned bid = get_group_id(0);
  unsigned tid = get_local_id(0);
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
  unsigned skip = skip_p[0];
  unsigned n = n_p[0];
  unsigned bid = get_group_id(0);
  unsigned tid = get_local_id(0);
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
  unsigned skip = skip_p[0];
  unsigned n = n_p[0];
  unsigned bid = get_group_id(0);
  unsigned tid = get_local_id(0);
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
  ss << R"EOS(
kernel void set_identity_kernel(constant unsigned *size_p, constant unsigned *skip_p, global float *py) {
  unsigned i = get_global_id(0);
  unsigned size = size_p[0];
  unsigned skip = skip_p[0];
  if (i < size) py[i] = !(i % skip);
}
)EOS";
  ss << R"EOS(
kernel void pick_fw_kernel(constant float *px, constant unsigned *pi,
                           constant unsigned *wx_p, constant unsigned *wy_p, constant unsigned *sx_p,
                           constant unsigned *si_p, constant unsigned *sy_p, global float *py) {
  unsigned wx = wx_p[0];
  unsigned wy = wy_p[0];
  unsigned sx = sx_p[0];
  unsigned si = si_p[0];
  unsigned sy = sy_p[0];
  unsigned t = get_global_id(0);
  unsigned bid_y = get_group_id(1);
  unsigned ox = bid_y * sx + pi[bid_y * si] * wy;
  unsigned oy = bid_y * sy;
  if (t < sy) py[oy + t] = px[ox + (t / wy) * wx + (t % wy)];
}
)EOS";
  ss << R"EOS(
kernel void slice_fw_kernel(constant float *px, constant unsigned *shift_p, constant unsigned *span_p,
                            constant unsigned *skip_p, constant unsigned *size_p, global float *py) {
  unsigned span = span_p[0];
  unsigned skip = skip_p[0];
  unsigned size = size_p[0];
  unsigned i = get_global_id(0);
  if (i < size) py[i] = px[(i / span) * skip + (i % span) + shift_p[0]];
}
)EOS";
  ss << R"EOS(
kernel void concat_fw_kernel(constant float *px, constant unsigned *span_p, constant unsigned *skip_p,
                             constant unsigned *x_size_p, constant unsigned *y_size_p, global float *py, constant unsigned *shift_p) {
  unsigned span = span_p[0];
  unsigned skip = skip_p[0];
  unsigned x_size = x_size_p[0];
  unsigned y_size = y_size_p[0];
  unsigned i = get_global_id(0);
  if (i < y_size) py[(i / span) * skip + (i % span) + shift_p[0]] = px[i % x_size];
}
)EOS";
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
  if (program.build({device_}, "-cl-std=CL2.0 -Werror") != CL_SUCCESS) {
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
  set_identity_kernel_ = cl::Kernel(program, "set_identity_kernel", &error);
  pick_fw_kernel_ = cl::Kernel(program, "pick_fw_kernel", &error);
  slice_fw_kernel_ = cl::Kernel(program, "slice_fw_kernel", &error);
  concat_fw_kernel_ = cl::Kernel(program, "concat_fw_kernel", &error);
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
  queue.enqueueReadBuffer(CDATA(x), CL_TRUE, 0,
            sizeof(cl_float) * num_elements, ret.data(), NULL, NULL);
  return ret;
}

std::vector<std::uint32_t> OpenCL::argmax_impl(const Tensor &x, std::uint32_t dim) {
  cl_int error = CL_SUCCESS;
  const Shape &shape = x.shape();
  std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  std::uint32_t s = shape.lower_volume(dim);
  std::uint32_t group_size = argmax_kernel_[0].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  while (group_size >> 1 >= n) group_size >>= 1;
  cl::CommandQueue queue(context_, device_, 0, &error);
  cl::Buffer mem_s = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(cl_uint), &s, &error);
  cl::Buffer mem_n = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(cl_uint), &n, &error);
  cl::Buffer py = cl::Buffer(context_,
      CL_MEM_WRITE_ONLY,
      sizeof(cl_uint) * r, NULL, &error);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      argmax_kernel_[m].setArg(0, CDATA(x)); \
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
            sizeof(cl_uint) * r, ret.data(), NULL, NULL);
  return ret;
}

std::vector<std::uint32_t> OpenCL::argmin_impl(const Tensor &x, std::uint32_t dim) {
  cl_int error = CL_SUCCESS;
  const Shape &shape = x.shape();
  std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  std::uint32_t s = shape.lower_volume(dim);
  std::uint32_t group_size = argmin_kernel_[0].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  while (group_size >> 1 >= n) group_size >>= 1;
  cl::CommandQueue queue(context_, device_, 0, &error);
  cl::Buffer mem_s = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(cl_uint), &s, &error);
  cl::Buffer mem_n = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(cl_uint), &n, &error);
  cl::Buffer py = cl::Buffer(context_,
      CL_MEM_WRITE_ONLY,
      sizeof(cl_uint) * r, NULL, &error);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      argmin_kernel_[m].setArg(0, CDATA(x)); \
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
            sizeof(cl_uint) * r, ret.data(), NULL, NULL);
  return ret;
}

void OpenCL::reset_tensor_impl(float k, Tensor &x) {
  std::int32_t error = CL_SUCCESS;
  const std::uint32_t size = x.shape().size();
  cl::CommandQueue queue(context_, device_, 0, &error);
  queue.enqueueFillBuffer<float>(CDATA(x), k, 0, sizeof(float) * size);
  queue.finish();
}

void OpenCL::reset_tensor_by_array_impl(const float values[], Tensor &x) {
  std::int32_t error = CL_SUCCESS;
  const std::uint32_t size = x.shape().size();
  cl::CommandQueue queue(context_, device_, 0, &error);
  queue.enqueueWriteBuffer(CDATA(x), CL_TRUE, 0,
            sizeof(float) * size, values, NULL, NULL);
}

void OpenCL::copy_tensor_impl(const Tensor &x, Tensor &y) {
  switch (x.device().type()) {
    case Device::DEVICE_TYPE_CPU:
      reset_tensor_by_array(static_cast<const float *>((x).data()), y);
      break;
    case Device::DEVICE_TYPE_OPENCL:
      {
        std::int32_t error = CL_SUCCESS;
        const std::uint32_t size = x.shape().size();
        cl::CommandQueue queue(context_, device_, 0, &error);
        queue.enqueueCopyBuffer(CDATA(x), CDATA(y), 0, 0, sizeof(float) * size);
        queue.finish();
      }
      break;
    default:
      reset_tensor_by_vector(x.to_vector(), y);
  }
}

void OpenCL::identity_impl(Tensor &y) {
  cl_int error = CL_SUCCESS;
  std::uint32_t size = y.shape().size();
  std::uint32_t skip = y.shape()[0] + 1;
  const std::uint32_t group_size = set_identity_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  const std::uint32_t num_blocks = GRID_SIZE(size, group_size);
  cl::CommandQueue queue(context_, device_, 0, &error);
  SET_ARG_HOST_SCALAR(set_identity_kernel_, 0, cl_uint, size)
  SET_ARG_HOST_SCALAR(set_identity_kernel_, 1, cl_uint, skip)
  set_identity_kernel_.setArg(2, CDATA(y));
  queue.enqueueNDRangeKernel(set_identity_kernel_, cl::NullRange, cl::NDRange(num_blocks * group_size), cl::NDRange(group_size), NULL, NULL);
  queue.finish();
}

void OpenCL::pick_fw_impl(const Tensor &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim, Tensor &y) {
  cl_int error = CL_SUCCESS;
  std::uint32_t wy = y.shape().lower_volume(dim);
  std::uint32_t wx = wy * x.shape()[dim];
  std::uint32_t sx = x.shape().has_batch() * x.shape().volume();
  std::uint32_t si = ids.size() > 1;
  std::uint32_t sy = y.shape().volume();
  const std::uint32_t group_size = pick_fw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  const std::uint32_t g1 = GRID_SIZE(sy, group_size);
  const std::uint32_t bs = y.shape().batch();
  cl::CommandQueue queue(context_, device_, 0, &error);
  pick_fw_kernel_.setArg(0, CDATA(x));
  SET_ARG_HOST_VECTOR(pick_fw_kernel_, 1, cl_uint, ids)
  SET_ARG_HOST_SCALAR(pick_fw_kernel_, 2, cl_uint, wx)
  SET_ARG_HOST_SCALAR(pick_fw_kernel_, 3, cl_uint, wy)
  SET_ARG_HOST_SCALAR(pick_fw_kernel_, 4, cl_uint, sx)
  SET_ARG_HOST_SCALAR(pick_fw_kernel_, 5, cl_uint, si)
  SET_ARG_HOST_SCALAR(pick_fw_kernel_, 6, cl_uint, sy)
  pick_fw_kernel_.setArg(7, CDATA(y));
  queue.enqueueNDRangeKernel(pick_fw_kernel_, cl::NullRange, cl::NDRange(g1 * group_size, bs), cl::NDRange(group_size, 1), NULL, NULL);
  queue.finish();
}

void OpenCL::slice_fw_impl(const Tensor &x, std::uint32_t dim, std::uint32_t offset, Tensor &y) {
  cl_int error = CL_SUCCESS;
  const std::uint32_t base = y.shape().lower_volume(dim);
  std::uint32_t shift = base * offset;
  std::uint32_t span = base * y.shape()[dim];
  std::uint32_t skip = base * x.shape()[dim];
  std::uint32_t size = y.shape().size();
  const std::uint32_t group_size = slice_fw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  const std::uint32_t num_blocks = GRID_SIZE(size, group_size);
  cl::CommandQueue queue(context_, device_, 0, &error);
  slice_fw_kernel_.setArg(0, CDATA(x));
  SET_ARG_HOST_SCALAR(slice_fw_kernel_, 1, cl_uint, shift)
  SET_ARG_HOST_SCALAR(slice_fw_kernel_, 2, cl_uint, span)
  SET_ARG_HOST_SCALAR(slice_fw_kernel_, 3, cl_uint, skip)
  SET_ARG_HOST_SCALAR(slice_fw_kernel_, 4, cl_uint, size)
  slice_fw_kernel_.setArg(5, CDATA(y));
  queue.enqueueNDRangeKernel(slice_fw_kernel_, cl::NullRange, cl::NDRange(num_blocks * group_size), cl::NDRange(group_size), NULL, NULL);
  queue.finish();
}

void OpenCL::concat_fw_impl(const std::vector<const Tensor *> &xs, std::uint32_t dim, Tensor &y) {
  cl_int error = CL_SUCCESS;
  const std::uint32_t new_bs = y.shape().batch();
  const std::uint32_t base = y.shape().lower_volume(dim);
  std::uint32_t skip = base * y.shape()[dim];
  std::uint32_t repeat = y.shape().volume() / skip;
  cl::CommandQueue queue(context_, device_, 0, &error);
  std::uint32_t offset = 0;
  const std::uint32_t group_size = concat_fw_kernel_.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  for (const Tensor *x : xs) {
    std::uint32_t span = base * x->shape()[dim];
    std::uint32_t x_size = span * repeat * x->shape().batch();
    std::uint32_t y_size = span * repeat * new_bs;
    const std::uint32_t num_blocks = GRID_SIZE(y_size, group_size);
    concat_fw_kernel_.setArg(0, CDATA(*x));
    SET_ARG_HOST_SCALAR(concat_fw_kernel_, 1, cl_uint, span)
    SET_ARG_HOST_SCALAR(concat_fw_kernel_, 2, cl_uint, skip)
    SET_ARG_HOST_SCALAR(concat_fw_kernel_, 3, cl_uint, x_size)
    SET_ARG_HOST_SCALAR(concat_fw_kernel_, 4, cl_uint, y_size)
    concat_fw_kernel_.setArg(5, CDATA(y));
    SET_ARG_HOST_SCALAR(concat_fw_kernel_, 6, cl_uint, offset)
    queue.enqueueNDRangeKernel(concat_fw_kernel_, cl::NullRange, cl::NDRange(num_blocks * group_size), cl::NDRange(group_size), NULL, NULL);
    offset += span;
  }
  queue.finish();
}

void OpenCL::sum_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  cl_int error = CL_SUCCESS;
  std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t group_size = sum_fw_kernel_[0].getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device_);
  while (group_size >> 1 >= n) group_size >>= 1;
  cl::CommandQueue queue(context_, device_, 0, &error);
  cl::Buffer mem_s = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(cl_uint), &s, &error);
  cl::Buffer mem_n = cl::Buffer(context_,
      CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
      sizeof(cl_uint), &n, &error);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      sum_fw_kernel_[m].setArg(0, CDATA(x)); \
      sum_fw_kernel_[m].setArg(1, mem_s); \
      sum_fw_kernel_[m].setArg(2, mem_n); \
      sum_fw_kernel_[m].setArg(3, CDATA(y)); \
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
