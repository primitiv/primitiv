#include <config.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <primitiv/cuda_device.h>

using std::cerr;
using std::endl;

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::stringstream ss; \
    ss << "CUDA function failed. statement: " << #f \
       << ", error: [" << err \
       << "] " << ::cudaGetErrorString(err); \
    throw std::runtime_error(ss.str()); \
  } \
}

namespace {

/*
 * CUDA kernels
 */

#define IDX threadIdx.x + blockIdx.x * blockDim.x

__global__ void dev_set_const(float *py, const float k, const unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = k;
}

__global__ void dev_negate(float *py, const float *px, const unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = -px[i];
}

__global__ void dev_add_const(
    float *py, const float *px, const float k, const unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = px[i] + k;
}

__global__ void dev_subtract_const_l(
    float *py, const float *px, const float k, const unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = k - px[i];
}

__global__ void dev_subtract_const_r(
    float *py, const float *px, const float k, const unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = px[i] - k;
}

__global__ void dev_multiply_const(
    float *py, const float *px, const float k, const unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = px[i] * k;
}

__global__ void dev_divide_const_l(
    float *py, const float *px, const float k, const unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = k / px[i];
}

__global__ void dev_divide_const_r(
    float *py, const float *px, const float k, const unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = px[i] / k;
}

__global__ void dev_exp(float *py, const float *px, const unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = ::expf(px[i]);
}

__global__ void dev_tanh(float *py, const float *px, const unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = ::tanhf(px[i]);
}

__global__ void dev_sigmoid(float *py, const float *px, const unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = .5f + .5f * ::tanhf(.5f * px[i]);
}

__global__ void dev_step(float *py, const float *px, const unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = (float)(px[i] >= .0f);
}

__global__ void dev_relu(float *py, const float *px, const unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = ::fmaxf(px[i], .0f);
}

#undef IDX

}  // namespace

namespace primitiv {

CUDADevice::CUDADevice(unsigned device_id)
: dev_id_(device_id) {
  int max_devs;
  CUDA_CALL(::cudaGetDeviceCount(&max_devs));
  if (dev_id_ >= static_cast<unsigned>(max_devs)) {
    std::stringstream ss;
    ss << "Invalid CUDA device ID. given: " << dev_id_ << " >= " << max_devs;
    throw std::runtime_error(ss.str());
  }
  CUDA_CALL(::cudaGetDeviceProperties(&prop_, dev_id_));
  // Dump device properties.
  cerr << "Selected CUDA Device " << dev_id_ << ':' << endl;
  cerr << "  Name ............ " << prop_.name << endl;
  cerr << "  Global Memory ... " << prop_.totalGlobalMem << endl;
  cerr << "  Shared Memory ... " << prop_.sharedMemPerBlock << endl;
  cerr << "  Threads/block ... " << prop_.maxThreadsPerBlock << endl;
  cerr << "  Threads dim ..... " << prop_.maxThreadsDim[0] << ','
                                 << prop_.maxThreadsDim[1] << ','
                                 << prop_.maxThreadsDim[2] << endl;
  cerr << "  Grid size ....... " << prop_.maxGridSize[0] << ','
                                 << prop_.maxGridSize[1] << ','
                                 << prop_.maxGridSize[2] << endl;
  max_threads_ = prop_.maxThreadsPerBlock;  // shortcut
}

CUDADevice::~CUDADevice() {
  // check memory leak
  if (!blocks_.empty()) {
    cerr << "FATAL ERROR: Detected memory leak on CUDADevice!" << endl;
    cerr << "Leaked blocks (handle: size):" << endl;
    for (const auto &kv : blocks_) {
      cerr << "  " << kv.first << ": " << kv.second << endl;
    }
    std::abort();
  }
}

Tensor CUDADevice::new_tensor(const Shape &shape) {
  const unsigned mem_size = sizeof(float) * shape.size();
  void *data;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMalloc(&data, mem_size));
  blocks_.insert(std::make_pair(data, mem_size));
  return Tensor(shape, this, data);
}

Tensor CUDADevice::new_tensor(const Shape &shape, const float k) {
  Tensor ret = new_tensor(shape);
  reset_tensor(ret, k);
  return ret;
}

Tensor CUDADevice::new_tensor(
    const Shape &shape, const std::vector<float> &values) {
  Tensor ret = new_tensor(shape);
  reset_tensor(ret, values);
  return ret;
}

void CUDADevice::delete_tensor(Tensor &x) {
  void *data = x.data();
  auto it = blocks_.find(data);
  if (it == blocks_.end()) {
    std::stringstream ss;
    ss << "Attempted to dispose unknown memory block: " << data;
    throw std::runtime_error(ss.str());
  }
  blocks_.erase(it);
  CUDA_CALL(::cudaFree(data));
}

#define CHECK_DEVICE(x) \
  if ((x).device() != this) { \
    std::stringstream ss; \
    ss << "Device mismatched. (" #x ").device(): " << (x).device() \
       << "!= this:" << this; \
    throw std::runtime_error(ss.str()); \
  }

#define DATA(x) static_cast<float *>((x).data())
#define CDATA(x) static_cast<const float *>((x).data())

std::vector<float> CUDADevice::tensor_to_vector(const Tensor &x) {
  CHECK_DEVICE(x);
  const unsigned num_elements = x.shape().size();
  std::vector<float> ret(num_elements);
  CUDA_CALL(::cudaMemcpy(
        &ret[0], x.data(), sizeof(float) * num_elements,
        cudaMemcpyDeviceToHost));
  return ret;
}

void CUDADevice::reset_tensor(Tensor &x, const float k) {
  CHECK_DEVICE(x);
  const unsigned num_elements = x.shape().size();
  const unsigned num_blocks = (num_elements + max_threads_ - 1) / max_threads_;
  ::dev_set_const<<<num_blocks, max_threads_>>>(DATA(x), k, num_elements);
}

void CUDADevice::reset_tensor(Tensor &x, const std::vector<float> &values) {
  CHECK_DEVICE(x);
  const unsigned num_elements = x.shape().size();
  if (values.size() != num_elements) {
    std::stringstream ss;
    ss << "Data sizes mismatched. required: " << num_elements
       << " (shape: " << x.shape().to_string() << ") != actual: "
       << values.size();
    throw std::runtime_error(ss.str());
  }
  CUDA_CALL(::cudaMemcpy(
        x.data(), &values[0], sizeof(float) * num_elements,
        cudaMemcpyHostToDevice));
}

Tensor CUDADevice::duplicate(const Tensor &x) {
  CHECK_DEVICE(x);
  Tensor ret = new_tensor(x.shape());
  ::cudaMemcpy(
      ret.data(), x.data(), sizeof(float) * x.shape().size(),
      cudaMemcpyDeviceToDevice);
  return ret;
}

#define CUDA_DEV_UNARY(name, kernel) \
Tensor CUDADevice::name(const Tensor &x) { \
  CHECK_DEVICE(x); \
  Tensor ret = new_tensor(x.shape()); \
  const unsigned size = x.shape().size(); \
  const unsigned num_blocks = (size + max_threads_ - 1) / max_threads_; \
  ::kernel<<<num_blocks, max_threads_>>>(DATA(ret), CDATA(x), size); \
  return ret; \
}

#define CUDA_DEV_BINARY_KX(name, kernel) \
Tensor CUDADevice::name(const float k, const Tensor &x) { \
  CHECK_DEVICE(x); \
  Tensor ret = new_tensor(x.shape()); \
  const unsigned size = x.shape().size(); \
  const unsigned num_blocks = (size + max_threads_ - 1) / max_threads_; \
  ::kernel<<<num_blocks, max_threads_>>>(DATA(ret), CDATA(x), k, size); \
  return ret; \
}

#define CUDA_DEV_BINARY_XK(name, kernel) \
Tensor CUDADevice::name(const Tensor &x, const float k) { \
  CHECK_DEVICE(x); \
  Tensor ret = new_tensor(x.shape()); \
  const unsigned size = x.shape().size(); \
  const unsigned num_blocks = (size + max_threads_ - 1) / max_threads_; \
  ::kernel<<<num_blocks, max_threads_>>>(DATA(ret), CDATA(x), k, size); \
  return ret; \
}

CUDA_DEV_UNARY(negate, dev_negate);
CUDA_DEV_UNARY(exp, dev_exp);
CUDA_DEV_UNARY(tanh, dev_tanh);
CUDA_DEV_UNARY(sigmoid, dev_sigmoid);
CUDA_DEV_UNARY(step, dev_step);
CUDA_DEV_UNARY(relu, dev_relu);

CUDA_DEV_BINARY_XK(add, dev_add_const);
CUDA_DEV_BINARY_KX(subtract, dev_subtract_const_l);
CUDA_DEV_BINARY_XK(subtract, dev_subtract_const_r);
CUDA_DEV_BINARY_XK(multiply, dev_multiply_const);
CUDA_DEV_BINARY_KX(divide, dev_divide_const_l);
CUDA_DEV_BINARY_XK(divide, dev_divide_const_r);

#undef CUDA_DEV_UNARY
#undef CUDA_DEV_BINARY_KX
#undef CUDA_DEV_BINARY_XK

Tensor CUDADevice::add(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::subtract(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::multiply(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::divide(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::transpose(const Tensor &x) {
  CHECK_DEVICE(x);
  const Shape &s = x.shape();
  if (s.dims().size() > 2) {
    std::stringstream ss;
    ss << "Attempted to transpose a tensor with shape " << s.to_string() << '.';
    throw std::runtime_error(ss.str());
  }
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::dot(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  throw std::runtime_error("not implemented.");
}

void CUDADevice::add_gradient(Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  throw std::runtime_error("not implemented.");
}

}  // namespace primitiv
