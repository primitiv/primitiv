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

#define IDX (threadIdx.x + blockIdx.x * blockDim.x)

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

__global__ void dev_add(
    float *py, const float *pa, const float *pb,
    const unsigned size, const unsigned mba, const unsigned mbb) {
  const unsigned i = IDX;
  const unsigned shift = blockIdx.y * size;
  if (i < size) py[i + shift] = pa[i + mba * shift] + pb[i + mbb * shift];
}

__global__ void dev_subtract(
    float *py, const float *pa, const float *pb,
    const unsigned size, const unsigned mba, const unsigned mbb) {
  const unsigned i = IDX;
  const unsigned shift = blockIdx.y * size;
  if (i < size) py[i + shift] = pa[i + mba * shift] - pb[i + mbb * shift];
}

__global__ void dev_multiply(
    float *py, const float *pa, const float *pb,
    const unsigned size, const unsigned mba, const unsigned mbb) {
  const unsigned i = IDX;
  const unsigned shift = blockIdx.y * size;
  if (i < size) py[i + shift] = pa[i + mba * shift] * pb[i + mbb * shift];
}

__global__ void dev_divide(
    float *py, const float *pa, const float *pb,
    const unsigned size, const unsigned mba, const unsigned mbb) {
  const unsigned i = IDX;
  const unsigned shift = blockIdx.y * size;
  if (i < size) py[i + shift] = pa[i + mba * shift] / pb[i + mbb * shift];
}

__global__ void dev_transpose(
    float *py, const float *px, const unsigned rows, const unsigned cols) {
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned j = threadIdx.y + blockIdx.y * blockDim.y;
  const unsigned ofs = blockIdx.z * rows * cols;
  if (i < rows && j < cols) {
    py[ofs + j + i * cols] = px[ofs + i + j * rows];
  }
}

__global__ void dev_dot(
    float *py, const float *pa, const float *pb,
    const unsigned di, const unsigned dj, const unsigned dk,
    const unsigned mba, const unsigned mbb) {
  // TODO(odashi): This implementation might be slow.
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned k = threadIdx.y + blockIdx.y * blockDim.y;
  if (i < di && k < dk) {
    pa += i + mba * blockIdx.z * di * dj;
    pb += (k + mbb * blockIdx.z * dk) * dj;
    float sum = .0f;
    for (unsigned j = 0; j < dj; ++j, pa += di, ++pb) sum += *pa * *pb;
    py[i + (k + blockIdx.z * dk) * di] = sum;
  }
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
  if (i < size) py[i] = (float)(px[i] > .0f);
}

__global__ void dev_relu(float *py, const float *px, const unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = ::fmaxf(px[i], .0f);
}

__global__ void dev_add_grad(
    float *pgx, const float *pgy, const unsigned size,
    const unsigned bs, const unsigned mbx, const unsigned mby) {
  // TODO(odashi): This implementation might be slow.
  const unsigned i = IDX;
  const unsigned shx = mbx * size;
  const unsigned shy = mby * size;
  if (i < size) {
    pgx += i, pgy += i;
    for (unsigned n = 0; n < bs; ++n, pgx += shx, pgy += shy) {
      *pgx += *pgy;
    }
  }
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
  // Calculates size of dims to be used in CUDA kernels.
  dim1_x_ = dim2_y_ = prop_.maxThreadsPerBlock;
  dim2_x_ = 1;
  while (dim2_x_ < dim2_y_) {
    dim2_x_ <<= 1;
    dim2_y_ >>= 1;
  }
  cerr << "Block configuration:" << endl;
  cerr << "  1 dim .... " << dim1_x_ << " threads" << endl;
  cerr << "  2 dims ... " << dim2_x_ << "x" << dim2_y_ << " threads" << endl;
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

#define GRID_SIZE(x, thread_size) ((x + thread_size - 1) / thread_size)
#define DATA(x) static_cast<float *>((x).data())
#define CDATA(x) static_cast<const float *>((x).data())

std::vector<float> CUDADevice::tensor_to_vector(const Tensor &x) {
  CHECK_DEVICE(x);
  const unsigned size = x.shape().size();
  std::vector<float> ret(size);
  CUDA_CALL(::cudaMemcpy(
        &ret[0], x.data(), sizeof(float) * size, cudaMemcpyDeviceToHost));
  return ret;
}

void CUDADevice::reset_tensor(Tensor &x, const float k) {
  CHECK_DEVICE(x);
  const unsigned size = x.shape().size();
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_);
  ::dev_set_const<<<num_blocks, dim1_x_>>>(DATA(x), k, size);
}

void CUDADevice::reset_tensor(Tensor &x, const std::vector<float> &values) {
  CHECK_DEVICE(x);
  const unsigned size = x.shape().size();
  if (values.size() != size) {
    std::stringstream ss;
    ss << "Data sizes mismatched. required: " << size
       << " (shape: " << x.shape().to_string() << ") != actual: "
       << values.size();
    throw std::runtime_error(ss.str());
  }
  CUDA_CALL(::cudaMemcpy(
        x.data(), &values[0], sizeof(float) * size, cudaMemcpyHostToDevice));
}

Tensor CUDADevice::random_uniform(
    const Shape &shape, const float lower, const float upper) {
  throw std::runtime_error("not implemented.");
}

Tensor CUDADevice::random_normal(
    const Shape &shape, const float mean, const float sd) {
  throw std::runtime_error("not implemented.");
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
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_); \
  ::kernel<<<num_blocks, dim1_x_>>>(DATA(ret), CDATA(x), size); \
  return ret; \
}

#define CUDA_DEV_BINARY_KX(name, kernel) \
Tensor CUDADevice::name(const float k, const Tensor &x) { \
  CHECK_DEVICE(x); \
  Tensor ret = new_tensor(x.shape()); \
  const unsigned size = x.shape().size(); \
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_); \
  ::kernel<<<num_blocks, dim1_x_>>>(DATA(ret), CDATA(x), k, size); \
  return ret; \
}

#define CUDA_DEV_BINARY_XK(name, kernel) \
Tensor CUDADevice::name(const Tensor &x, const float k) { \
  CHECK_DEVICE(x); \
  Tensor ret = new_tensor(x.shape()); \
  const unsigned size = x.shape().size(); \
  const unsigned num_blocks = GRID_SIZE(size,dim1_x_); \
  ::kernel<<<num_blocks, dim1_x_>>>(DATA(ret), CDATA(x), k, size); \
  return ret; \
}

#define CUDA_DEV_BINARY_AB(name, kernel) \
Tensor CUDADevice::name(const Tensor &a, const Tensor &b) { \
  CHECK_DEVICE(a); \
  CHECK_DEVICE(b); \
  const Shape &sa = a.shape(); \
  const Shape &sb = b.shape(); \
  const unsigned ba = sa.batch_size(); \
  const unsigned bb = sb.batch_size(); \
  const unsigned size = sa.size() / ba; \
  const unsigned x = GRID_SIZE(size, dim1_x_); \
  const unsigned y = std::max(ba, bb); \
  if (sa.dims() != sb.dims() || (ba != bb && ba > 1 && bb > 1)) { \
    std::stringstream ss; \
    ss << "Attempted to " #name " tensors with shapes " \
       << sa.to_string() << " and " << sb.to_string() << '.'; \
    throw std::runtime_error(ss.str()); \
  } \
  Tensor ret = new_tensor(Shape(sa.dims(), y)); \
  ::kernel<<<dim3(x, y, 1), dim1_x_>>>( \
      DATA(ret), CDATA(a), CDATA(b), size, ba > 1, bb > 1); \
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

CUDA_DEV_BINARY_AB(add, dev_add);
CUDA_DEV_BINARY_AB(subtract, dev_subtract);
CUDA_DEV_BINARY_AB(multiply, dev_multiply);
CUDA_DEV_BINARY_AB(divide, dev_divide);

#undef CUDA_DEV_UNARY
#undef CUDA_DEV_BINARY_KX
#undef CUDA_DEV_BINARY_XK
#undef CUDA_DEV_BINARY_AB

Tensor CUDADevice::transpose(const Tensor &x) {
  CHECK_DEVICE(x);
  const Shape &s = x.shape();
  const unsigned d1 = s.dim(0);
  const unsigned d2 = s.dim(1);
  const unsigned bs = s.batch_size();
  const unsigned g1 = GRID_SIZE(d1, dim2_x_);
  const unsigned g2 = GRID_SIZE(d2, dim2_y_);
  if (s.dims().size() > 2) {
    std::stringstream ss;
    ss << "Attempted to transpose a tensor with shape " << s.to_string() << '.';
    throw std::runtime_error(ss.str());
  }
  Tensor ret = new_tensor(Shape({d2, d1}, bs));
  ::dev_transpose<<<dim3(g1, g2, bs), dim3(dim2_x_, dim2_y_, 1)>>>(
      DATA(ret), CDATA(x), d1, d2);
  return ret;
}

Tensor CUDADevice::dot(const Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const unsigned di = sa.dim(0);
  const unsigned dj = sa.dim(1);
  const unsigned dk = sb.dim(1);
  const unsigned ba = sa.batch_size();
  const unsigned bb = sb.batch_size();
  const unsigned g1 = GRID_SIZE(di, dim2_x_);
  const unsigned g2 = GRID_SIZE(dk, dim2_y_);
  const unsigned bs = std::max(ba, bb);
  if (sa.dims().size() > 2 || sb.dims().size() > 2 ||
      sb.dim(0) != dj ||
      (ba != bb && ba > 1 && bb > 1)) {
    std::stringstream ss;
    ss << "Attempted to calculate the dot product of tensors with shapes "
      << sa.to_string() << " and " << sb.to_string() << '.';
    throw std::runtime_error(ss.str());
  }
  Tensor ret = new_tensor(Shape({di, dk}, bs));
  ::dev_dot<<<dim3(g1, g2, bs), dim3(dim2_x_, dim2_y_, 1)>>>(
      DATA(ret), CDATA(a), CDATA(b), di, dj, dk, ba > 1, bb > 1);
  return ret;
}

void CUDADevice::add_gradient(Tensor &a, const Tensor &b) {
  CHECK_DEVICE(a);
  CHECK_DEVICE(b);
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const unsigned ba = sa.batch_size();
  const unsigned bb = sb.batch_size();
  const unsigned size = sa.size() / ba;
  const unsigned g1 = GRID_SIZE(size, dim1_x_);
  if (sa.dims() != sb.dims() || (ba != bb && ba > 1 && bb > 1)) {
    std::stringstream ss;
    ss << "Attempted to add gradients with shape " << sb.to_string()
       << " to shape " << sa.to_string() << '.';
    throw std::runtime_error(ss.str());
  }
  ::dev_add_grad<<<g1, dim1_x_>>>(
      DATA(a), CDATA(b), size, std::max(ba, bb), ba > 1, bb > 1);
}

}  // namespace primitiv
