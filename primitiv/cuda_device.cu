#include <config.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <cmath>
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

/**
 * Reset arrays with a constant.
 * @param ptr Target array.
 * @param k Constant.
 * @param size Number of elements in the array.
 */
__global__ void cuda_set_const(float *ptr, const float k, const unsigned size) {
  const unsigned pos = threadIdx.x + blockIdx.x * blockDim.x;
  if (pos < size) ptr[pos] = k;
}

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

#define DATA(x) static_cast<float *>((x).data());
#define CDATA(x) static_cast<const float *>((x).data());

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
  const unsigned num_blocks =
    (num_elements + prop_.maxThreadsPerBlock - 1) / prop_.maxThreadsPerBlock;
  ::cuda_set_const<<<num_blocks, 1024>>>(
      static_cast<float *>(x.data()), k, num_elements);
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
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::negate(const Tensor &x) {
  CHECK_DEVICE(x);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::add(const Tensor &x, const float k) {
  CHECK_DEVICE(x);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::add(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::subtract(const Tensor &x, const float k) {
  CHECK_DEVICE(x);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::subtract(const float k, const Tensor &x) {
  CHECK_DEVICE(x);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::subtract(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::multiply(const Tensor &x, const float k) {
  CHECK_DEVICE(x);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::multiply(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::divide(const Tensor &x, const float k) {
  CHECK_DEVICE(x);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::divide(const float k, const Tensor &x) {
  CHECK_DEVICE(x);
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

Tensor CUDADevice::exp(const Tensor &x) {
  CHECK_DEVICE(x);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::tanh(const Tensor &x) {
  CHECK_DEVICE(x);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::sigmoid(const Tensor &x) {
  CHECK_DEVICE(x);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::step(const Tensor &x) {
  CHECK_DEVICE(x);
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::relu(const Tensor &x) {
  CHECK_DEVICE(x);
  throw std::runtime_error("not implemented.");
}

void CUDADevice::add_gradient(Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  throw std::runtime_error("not implemented.");
}

}  // namespace primitiv
