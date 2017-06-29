#include <config.h>

#include <cuda_runtime_api.h>
#include <iostream>
#include <random>
#include <primitiv/cuda_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/error.h>

using std::cerr;
using std::endl;

namespace {

/*
 * CUDA kernels
 */

#define IDX (threadIdx.x + blockIdx.x * blockDim.x)

__global__ void set_const_dev(float *py, float k, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = k;
}

__global__ void rand_bernoulli_dev(float *px, float p, float size) {
  const unsigned i = IDX;
  if (i < size) px[i] = (float)(px[i] <= p);
}

__global__ void rand_affine_dev(
    float *px, float shift, float scale, unsigned size) {
  const unsigned i = IDX;
  if (i < size) px[i] = px[i] * scale + shift;
}

__global__ void slice_fw_dev(
    float *py, const float *px, unsigned span, unsigned skip, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = px[(i / span) * skip + (i % span)];
}

__global__ void concat_fw_dev(
    float *py, const float *px,
    unsigned span, unsigned skip, unsigned x_size, unsigned y_size) {
  const unsigned i = IDX;
  if (i < y_size) py[(i / span) * skip + (i % span)] = px[i % x_size];
}

#define CUDA_KERNEL_X(name, op) \
__global__ void name##_fw_dev(float *py, const float *px, unsigned size) { \
  const unsigned i = IDX; \
  if (i < size) py[i] = (op); \
}

#define CUDA_KERNEL_X_CONST(name, op) \
__global__ void name##_fw_dev(float *py, const float *px, float k, unsigned size) { \
  const unsigned i = IDX; \
  if (i < size) py[i] = (op); \
}

#define CUDA_KERNEL_X_SCALAR_R(name, op) \
__global__ void name##_fw_dev( \
    float *py, const float *px, const float *pk, \
    unsigned size, unsigned mbx, unsigned mbk) { \
  const unsigned i = IDX; \
  const unsigned shift = blockIdx.y * size; \
  if (i < size) py[i + shift] = op(px[i + mbx * shift], pk[mbk * blockIdx.y]); \
}

#define CUDA_KERNEL_X_SCALAR_L(name, op) \
__global__ void name##_fw_dev( \
    float *py, const float *px, const float *pk, \
    unsigned size, unsigned mbx, unsigned mbk) { \
  const unsigned i = IDX; \
  const unsigned shift = blockIdx.y * size; \
  if (i < size) py[i + shift] = op(pk[mbk * blockIdx.y], px[i + mbx * shift]); \
}

#define CUDA_KERNEL_AB(name, op) \
__global__ void name##_fw_dev( \
    float *py, const float *pa, const float *pb, \
    unsigned size, unsigned mba, unsigned mbb) { \
  const unsigned i = IDX; \
  const unsigned shift = blockIdx.y * size; \
  if (i < size) py[i + shift] = op(pa[i + mba * shift], pb[i + mbb * shift]); \
}

CUDA_KERNEL_X(negate, -px[i]);
CUDA_KERNEL_X(sqrt, ::__fsqrt_rn(px[i]));
CUDA_KERNEL_X(exp, ::expf(px[i]));
CUDA_KERNEL_X(tanh, ::tanhf(px[i]));
CUDA_KERNEL_X(sigmoid, .5f + .5f * ::tanhf(.5f * px[i]));
CUDA_KERNEL_X(sin, ::sinf(px[i]));
CUDA_KERNEL_X(cos, ::cosf(px[i]));
CUDA_KERNEL_X(tan, ::tanf(px[i]));

CUDA_KERNEL_X_CONST(add_const, px[i] + k);
CUDA_KERNEL_X_CONST(subtract_const_r, px[i] - k);
CUDA_KERNEL_X_CONST(subtract_const_l, k - px[i]);
CUDA_KERNEL_X_CONST(multiply_const, px[i] * k);
CUDA_KERNEL_X_CONST(divide_const_r, px[i] / k);
CUDA_KERNEL_X_CONST(divide_const_l, k / px[i]);
CUDA_KERNEL_X_CONST(pstep, (px[i] > .0f) + k * (px[i] <= .0f));
CUDA_KERNEL_X_CONST(prelu, px[i] * ((px[i] > .0f) + k * (px[i] <= .0f)));

CUDA_KERNEL_X_SCALAR_R(add_scalar, ::__fadd_rn);
CUDA_KERNEL_X_SCALAR_R(subtract_scalar_r, ::__fsub_rn);
CUDA_KERNEL_X_SCALAR_L(subtract_scalar_l, ::__fsub_rn);
CUDA_KERNEL_X_SCALAR_R(multiply_scalar, ::__fmul_rn);
CUDA_KERNEL_X_SCALAR_R(divide_scalar_r, ::__fdiv_rn);
CUDA_KERNEL_X_SCALAR_L(divide_scalar_l, ::__fdiv_rn);

CUDA_KERNEL_AB(add, ::__fadd_rn);
CUDA_KERNEL_AB(subtract, ::__fsub_rn);
CUDA_KERNEL_AB(multiply, ::__fmul_rn);
CUDA_KERNEL_AB(divide, ::__fdiv_rn);

#undef CUDA_KERNEL_X
#undef CUDA_KERNEL_X_CONST
#undef CUDA_KERNEL_X_SCALAR_R
#undef CUDA_KERNEL_X_SCALAR_L

__global__ void transpose_fw_dev(
    float *py, const float *px, unsigned rows, unsigned cols) {
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned j = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned ofs = blockIdx.z * rows * cols;
  if (i < rows && j < cols) {
    py[ofs + j + i * cols] = px[ofs + i + j * rows];
  }
}

template<unsigned BLOCK_SIZE>
__global__ void sum_fw_dev(float *py, const float *px, unsigned skip, unsigned n) {
  __shared__ float temp[BLOCK_SIZE];
  const unsigned bid = blockIdx.x;
  const unsigned tid = threadIdx.x;
  px += bid % skip + (bid / skip) * skip * n;
  temp[tid] = 0;
  for (unsigned i = tid; i < n; i += BLOCK_SIZE) temp[tid] += px[i * skip];
  __syncthreads();
#define REDUCE(k) \
  if (BLOCK_SIZE >= k << 1) { \
    if (tid < k) temp[tid] += temp[tid + k]; \
    __syncthreads(); \
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
}

__device__ float logsumexp2_fw_dev(float a, float b) {
  return a > b
    ? a + ::log(1.f + ::exp(b - a))
    : b + ::log(1.f + ::exp(a - b));
}

template<unsigned BLOCK_SIZE>
__global__ void logsumexp_fw_dev(
    float *py, const float *px, unsigned skip, unsigned n) {
  __shared__ float temp[BLOCK_SIZE];
  const unsigned bid = blockIdx.x;
  const unsigned tid = threadIdx.x;
  px += bid % skip + (bid / skip) * skip * n;
  temp[tid] = -1e38;  // NOTE(odashi): Near the minimum of the float.
  for (unsigned i = tid; i < n; i += BLOCK_SIZE) {
    temp[tid] = ::logsumexp2_fw_dev(temp[tid], px[i * skip]);
  }
  __syncthreads();
#define REDUCE(k) \
  if (BLOCK_SIZE >= k << 1) { \
    if (tid < k) temp[tid] = ::logsumexp2_fw_dev(temp[tid], temp[tid + k]); \
    __syncthreads(); \
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
}

__global__ void broadcast_fw_dev(
    float *py, const float *px, unsigned skip1, unsigned skip2, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = px[i % skip1 + (i / skip2) * skip1];
}

__global__ void batch_sum_fw_dev(
    float *py, const float *px, unsigned size, unsigned batch) {
  const unsigned i = IDX;
  if (i < size) {
    float temp = .0f;
    px += i;
    for (unsigned j = 0; j < batch; ++j, px += size) {
      temp += *px;
    }
    py[i] = temp;
  }
}

__global__ void add_grad_dev(
    float *pgx, const float *pgy, unsigned nx, unsigned ny) {
  const unsigned i = IDX;
  if (i < ::max(nx, ny)) ::atomicAdd(pgx + i % nx, pgy[i % ny]);
}

__global__ void add_grad_ofs_dev(
    float *pgx, const float *pgy,
    unsigned wx, unsigned wy, unsigned nx, unsigned ny) {
  const unsigned i = IDX;
  if (i < wy * ::max(nx, ny)) {
    ::atomicAdd(
        pgx + ((i / wy) * wx + (i % wy)) % (wx * nx),
        pgy[i % (wy * ny)]);
  }
}

__global__ void add_grad_sparse_dev(
    float *pgx, const float *pgy, unsigned wx, unsigned wy, unsigned repeat) {
  const unsigned i = IDX;
  if (i < wy * repeat) {
    ::atomicAdd(pgx + (i / wy) * wx + (i % wy), pgy[i]);
  }
}

#undef IDX

}  // namespace

namespace {

// Minimum requirements of the compute capability.
static const int MIN_CC_MAJOR = 3;
static const int MIN_CC_MINOR = 0;

}

namespace primitiv {

unsigned CUDADevice::num_devices() {
  int ret;
  CUDA_CALL(::cudaGetDeviceCount(&ret));
  return ret;
}

void CUDADevice::initialize() {
  // Retrieves device properties.
  ::cudaDeviceProp prop;
  CUDA_CALL(::cudaGetDeviceProperties(&prop, dev_id_));

  // Dump device properties.
  cerr << "Selected CUDA Device " << dev_id_ << ':' << endl;
  cerr << "  Name ................. " << prop.name << endl;
  cerr << "  Global Memory ........ " << prop.totalGlobalMem << endl;
  cerr << "  Shared Memory ........ " << prop.sharedMemPerBlock << endl;
  cerr << "  Threads/block ........ " << prop.maxThreadsPerBlock << endl;
  cerr << "  Threads dim .......... " << prop.maxThreadsDim[0] << ','
                                      << prop.maxThreadsDim[1] << ','
                                      << prop.maxThreadsDim[2] << endl;
  cerr << "  Grid size ............ " << prop.maxGridSize[0] << ','
                                      << prop.maxGridSize[1] << ','
                                      << prop.maxGridSize[2] << endl;
  cerr << "  Compute Capability ... " << prop.major << '.'
                                      << prop.minor << endl;

  // Check compute capability requirements.
  if (prop.major < ::MIN_CC_MAJOR ||
      (prop.major == ::MIN_CC_MAJOR && prop.minor < ::MIN_CC_MINOR)) {
    THROW_ERROR(
        "CUDA Device " << dev_id_ << " does not satisfy the "
        "minimum requirement of the compute capability: "
        << prop.major << '.' << prop.minor << " < "
        << ::MIN_CC_MAJOR << '.' << ::MIN_CC_MINOR);
  }

  // Calculates size of dims to be used in CUDA kernels.
  dim1_x_ = 1;
  while (dim1_x_ < 1024 &&
      dim1_x_ < static_cast<unsigned>(prop.maxThreadsPerBlock)) {
    dim1_x_ <<= 1;
  }
  dim2_y_ = dim1_x_;
  dim2_x_ = 1;
  while (dim2_x_ < dim2_y_) {
    dim2_x_ <<= 1;
    dim2_y_ >>= 1;
  }
  cerr << "Block configuration:" << endl;
  cerr << "  1 dim .... " << dim1_x_ << " threads" << endl;
  cerr << "  2 dims ... " << dim2_x_ << "x" << dim2_y_ << " threads" << endl;

  // Initializes additional libraries
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUBLAS_CALL(::cublasCreate(&cublas_));
  CURAND_CALL(::curandCreateGenerator(&curand_, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(::curandSetPseudoRandomGeneratorSeed(curand_, rng_seed_));
}

CUDADevice::CUDADevice(unsigned device_id)
: dev_id_(device_id)
, rng_seed_(std::random_device()())
, pool_(device_id) {
  initialize();
}

CUDADevice::CUDADevice(unsigned device_id, unsigned rng_seed)
: dev_id_(device_id)
, rng_seed_(rng_seed)
, pool_(device_id) {
  initialize();
}

CUDADevice::~CUDADevice() {
  // Finalizes additional libraries
  CUBLAS_CALL(::cublasDestroy(cublas_));
  CURAND_CALL(::curandDestroyGenerator(curand_));
}

std::shared_ptr<void> CUDADevice::new_handle(const Shape &shape) {
  return pool_.allocate(sizeof(float) * shape.size());
}

#define GRID_SIZE(x, threads) (((x) + (threads) - 1) / (threads))
#define DATA(x) static_cast<float *>((x).data())
#define CDATA(x) static_cast<const float *>((x).data())

std::vector<float> CUDADevice::tensor_to_vector_impl(const Tensor &x) {
  const unsigned size = x.shape().size();
  std::vector<float> ret(size);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        &ret[0], x.data(), sizeof(float) * size, cudaMemcpyDeviceToHost));
  return ret;
}

void CUDADevice::reset_tensor_impl(Tensor &x, float k) {
  const unsigned size = x.shape().size();
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::set_const_dev<<<num_blocks, dim1_x_>>>(DATA(x), k, size);
}

void CUDADevice::reset_tensor_by_array_impl(Tensor &x, const float values[]) {
  const unsigned size = x.shape().size();
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        x.data(), values, sizeof(float) * size, cudaMemcpyHostToDevice));
}

void CUDADevice::copy_tensor_impl(const Tensor &x, Tensor &y) {
  switch (x.device()->type()) {
    case Device::DEVICE_TYPE_CPU:
      reset_tensor_by_array(y, CDATA(x));
      break;
    case Device::DEVICE_TYPE_CUDA:
      CUDA_CALL(::cudaSetDevice(dev_id_));
      CUDA_CALL(::cudaMemcpy(
            DATA(y), CDATA(x),
            sizeof(float) * x.shape().size(),
            cudaMemcpyDeviceToDevice));
      break;
    default:
      reset_tensor_by_vector(y, x.to_vector());
  }
}

void CUDADevice::random_bernoulli_impl(float p, Tensor &y) {
  const unsigned size = y.shape().size();
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateUniform(curand_, DATA(y), size));
  ::rand_bernoulli_dev<<<num_blocks, dim1_x_>>>(DATA(y), p, size);
}

void CUDADevice::random_uniform_impl(float lower, float upper, Tensor &y) {
  const unsigned size = y.shape().size();
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_);
  const float scale = upper - lower;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateUniform(curand_, DATA(y), size));
  ::rand_affine_dev<<<num_blocks, dim1_x_>>>(DATA(y), lower, scale, size);
}

void CUDADevice::random_normal_impl(float mean, float sd, Tensor &y) {
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateNormal(
        curand_, DATA(y), y.shape().size(), mean, sd));
}

void CUDADevice::random_log_normal_impl(float mean, float sd, Tensor &y) {
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateLogNormal(
        curand_, DATA(y), y.shape().size(), mean, sd));
}

Tensor CUDADevice::pick_fw_impl(
    const Tensor &x, unsigned dim,
    const std::vector<unsigned> &ids, Shape &&new_shape) {
  const unsigned base = new_shape.lower_volume(dim);
  const unsigned skip = base * x.shape()[dim];
  const unsigned size = new_shape.volume();
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_);
  const unsigned skip_x =
    (x.shape().has_batch()) * x.shape().volume();
  const unsigned skip_i = ids.size() > 1;
  Tensor ret = new_tensor(new_shape);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  for (unsigned b = 0; b < new_shape.batch(); ++b) {
    ::slice_fw_dev<<<num_blocks, dim1_x_>>>(
        DATA(ret) + b * size, CDATA(x) + b * skip_x + base * ids[b * skip_i],
        base, skip, size);
  }
  return ret;
}

Tensor CUDADevice::slice_fw_impl(
    const Tensor &x, unsigned dim, unsigned offset, Shape &&new_shape) {
  const unsigned base = new_shape.lower_volume(dim);
  const unsigned span = base * new_shape[dim];
  const unsigned skip = base * x.shape()[dim];
  const unsigned size = new_shape.size();
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_);
  Tensor ret = new_tensor(new_shape);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::slice_fw_dev<<<num_blocks, dim1_x_>>>(
      DATA(ret), CDATA(x) + base * offset, span, skip, size);
  return ret;
}

Tensor CUDADevice::concat_fw_impl(
    const std::vector<const Tensor *> &xs, unsigned dim, Shape &&new_shape) {
  const unsigned new_bs = new_shape.batch();
  const unsigned base = new_shape.lower_volume(dim);
  const unsigned skip = base * new_shape[dim];
  unsigned repeat = new_shape.volume() / skip;
  Tensor ret = new_tensor(new_shape);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  unsigned offset = 0;
  for (const Tensor *x : xs) {
    const unsigned span = base * x->shape()[dim];
    const unsigned x_size = span * repeat * x->shape().batch();
    const unsigned y_size = span * repeat * new_bs;
    const unsigned num_blocks = GRID_SIZE(y_size, dim1_x_);
    ::concat_fw_dev<<<num_blocks, dim1_x_>>>(
        DATA(ret) + offset, CDATA(*x), span, skip, x_size, y_size);
    offset += span;
  }
  return ret;
}

#define CUDA_DEV_UNARY(name) \
Tensor CUDADevice::name##_fw_impl(const Tensor &x) { \
  Tensor ret = new_tensor(x.shape()); \
  const unsigned size = x.shape().size(); \
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<num_blocks, dim1_x_>>>(DATA(ret), CDATA(x), size); \
  return ret; \
}

#define CUDA_DEV_BINARY_CONST(name) \
Tensor CUDADevice::name##_fw_impl(const Tensor &x, float k) { \
  Tensor ret = new_tensor(x.shape()); \
  const unsigned size = x.shape().size(); \
  const unsigned num_blocks = GRID_SIZE(size,dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<num_blocks, dim1_x_>>>(DATA(ret), CDATA(x), k, size); \
  return ret; \
}

#define CUDA_DEV_BINARY_SCALAR(name) \
Tensor CUDADevice::name##_fw_impl( \
    const Tensor &x, const Tensor &k, Shape &&new_shape) { \
  const unsigned size = new_shape.volume(); \
  const unsigned g1 = GRID_SIZE(size, dim1_x_); \
  const unsigned g2 = new_shape.batch(); \
  Tensor ret = new_tensor(new_shape); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<dim3(g1, g2, 1), dim1_x_>>>( \
      DATA(ret), CDATA(x), CDATA(k), size, \
      x.shape().has_batch(), k.shape().has_batch()); \
  return ret; \
}

#define CUDA_DEV_BINARY_AB(name) \
Tensor CUDADevice::name##_fw_impl( \
    const Tensor &a, const Tensor &b, Shape &&new_shape) { \
  const unsigned size = new_shape.volume(); \
  const unsigned x = GRID_SIZE(size, dim1_x_); \
  const unsigned y = new_shape.batch(); \
  Tensor ret = new_tensor(new_shape); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<dim3(x, y, 1), dim1_x_>>>( \
      DATA(ret), CDATA(a), CDATA(b), size, \
      a.shape().has_batch(), b.shape().has_batch()); \
  return ret; \
}

CUDA_DEV_UNARY(negate);
CUDA_DEV_UNARY(sqrt);
CUDA_DEV_UNARY(exp);
CUDA_DEV_UNARY(tanh);
CUDA_DEV_UNARY(sigmoid);
CUDA_DEV_UNARY(sin);
CUDA_DEV_UNARY(cos);
CUDA_DEV_UNARY(tan);

CUDA_DEV_BINARY_CONST(add_const);
CUDA_DEV_BINARY_CONST(subtract_const_r);
CUDA_DEV_BINARY_CONST(subtract_const_l);
CUDA_DEV_BINARY_CONST(multiply_const);
CUDA_DEV_BINARY_CONST(divide_const_r);
CUDA_DEV_BINARY_CONST(divide_const_l);
CUDA_DEV_BINARY_CONST(pstep);
CUDA_DEV_BINARY_CONST(prelu);

CUDA_DEV_BINARY_SCALAR(add_scalar);
CUDA_DEV_BINARY_SCALAR(subtract_scalar_r);
CUDA_DEV_BINARY_SCALAR(subtract_scalar_l);
CUDA_DEV_BINARY_SCALAR(multiply_scalar);
CUDA_DEV_BINARY_SCALAR(divide_scalar_r);
CUDA_DEV_BINARY_SCALAR(divide_scalar_l);

CUDA_DEV_BINARY_AB(add);
CUDA_DEV_BINARY_AB(subtract);
CUDA_DEV_BINARY_AB(multiply);
CUDA_DEV_BINARY_AB(divide);

#undef CUDA_DEV_UNARY
#undef CUDA_DEV_BINARY_CONST
#undef CUDA_DEV_BINARY_SCALAR
#undef CUDA_DEV_BINARY_AB

Tensor CUDADevice::transpose_fw_impl(const Tensor &x, Shape &&new_shape) {
  const unsigned d1 = new_shape[1];
  const unsigned d2 = new_shape[0];
  const unsigned bs = new_shape.batch();
  const unsigned g1 = GRID_SIZE(d1, dim2_x_);
  const unsigned g2 = GRID_SIZE(d2, dim2_y_);
  Tensor ret = new_tensor(new_shape);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::transpose_fw_dev<<<dim3(g1, g2, bs), dim3(dim2_x_, dim2_y_, 1)>>>(
      DATA(ret), CDATA(x), d1, d2);
  return ret;
}

Tensor CUDADevice::matmul_fw_impl(
    const Tensor &a, const Tensor &b, Shape &&new_shape) {
  const unsigned di = new_shape[0];
  const unsigned dj = a.shape()[1];
  const unsigned dk = new_shape[1];
  float alpha = 1.;
  float beta = 0.;
  Tensor ret = new_tensor(new_shape);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  if (a.shape().has_batch()) {
    // Do gemm multiple times.
    const unsigned a_skip = di * dj;
    const unsigned b_skip = b.shape().has_batch() * dj * dk;
    const unsigned y_skip = di * dk;
    const unsigned bs = a.shape().batch();
    for (unsigned n = 0; n < bs; ++n) {
      CUBLAS_CALL(::cublasSgemm(
            cublas_, ::CUBLAS_OP_N, ::CUBLAS_OP_N,
            di, dk, dj,
            &alpha, CDATA(a) + n * a_skip, di, CDATA(b) + n * b_skip, dj,
            &beta, DATA(ret) + n * y_skip, di));
    }
  } else {
    // Do gemm only once to calculate the product with a combined matrix.
    CUBLAS_CALL(::cublasSgemm(
          cublas_, ::CUBLAS_OP_N, ::CUBLAS_OP_N,
          di, dk * b.shape().batch(), dj,
          &alpha, CDATA(a), di, CDATA(b), dj,
          &beta, DATA(ret), di));
  }
  return ret;
}

void CUDADevice::matmul_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  // ga += gy . b^T
  // gb += a^T . gy
  const unsigned di = a.shape()[0];
  const unsigned dj = a.shape()[1];
  const unsigned dk = b.shape()[1];
  float alpha = 1.;
  float beta = 1.;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  if (a.shape().has_batch()) {
    // Do gemm multiple times.
    const unsigned a_skip = di * dj;
    const unsigned b_skip = b.shape().has_batch() * dj * dk;
    const unsigned y_skip = di * dk;
    const unsigned bs = a.shape().batch();
    for (unsigned n = 0; n < bs; ++n) {
      CUBLAS_CALL(::cublasSgemm(
            cublas_, ::CUBLAS_OP_N, ::CUBLAS_OP_T,
            di, dj, dk,
            &alpha, CDATA(gy) + n * y_skip, di, CDATA(b) + n * b_skip, dj,
            &beta, DATA(ga) + n * a_skip, di));
      CUBLAS_CALL(::cublasSgemm(
            cublas_, ::CUBLAS_OP_T, ::CUBLAS_OP_N,
            dj, dk, di,
            &alpha, CDATA(a) + n * a_skip, di, CDATA(gy) + n * y_skip, di,
            &beta, DATA(gb) + n * b_skip, dj));
    }
  } else {
    // Do gemm only once to calculate the product with a combined matrix.
    CUBLAS_CALL(::cublasSgemm(
          cublas_, ::CUBLAS_OP_N, ::CUBLAS_OP_T,
          di, dj, dk * b.shape().batch(),
          &alpha, CDATA(gy), di, CDATA(b), dj,
          &beta, DATA(ga), di));
    CUBLAS_CALL(::cublasSgemm(
          cublas_, ::CUBLAS_OP_T, ::CUBLAS_OP_N,
          dj, dk * b.shape().batch(), di,
          &alpha, CDATA(a), di, CDATA(gy), di,
          &beta, DATA(gb), dj));
  }
}

Tensor CUDADevice::sum_fw_impl(const Tensor &x, unsigned dim) {
  const Shape new_shape = x.shape().resize_dim(dim, 1);
  const unsigned n = x.shape()[dim];
  const unsigned r = new_shape.size();
  const unsigned s = new_shape.lower_volume(dim);
  unsigned block_size = dim1_x_;
  while (block_size >> 1 >= n) block_size >>= 1;
  Tensor ret = new_tensor(new_shape);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  switch (block_size) {
#define CASE(k) \
    case k: ::sum_fw_dev<k><<<r, k>>>(DATA(ret), CDATA(x), s, n); break
    CASE(1024);
    CASE(512);
    CASE(256);
    CASE(128);
    CASE(64);
    CASE(32);
    CASE(16);
    CASE(8);
    CASE(4);
    CASE(2);
    CASE(1);
#undef CASE
  }
  return ret;
}

Tensor CUDADevice::logsumexp_fw_impl(const Tensor &x, unsigned dim) {
  const Shape new_shape = x.shape().resize_dim(dim, 1);
  const unsigned n = x.shape()[dim];
  const unsigned r = new_shape.size();
  const unsigned s = new_shape.lower_volume(dim);
  unsigned block_size = dim1_x_;
  while (block_size >> 1 >= n) block_size >>= 1;
  Tensor ret = new_tensor(new_shape);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  switch (block_size) {
#define CASE(k) \
    case k: ::logsumexp_fw_dev<k><<<r, k>>>(DATA(ret), CDATA(x), s, n); break
    CASE(1024);
    CASE(512);
    CASE(256);
    CASE(128);
    CASE(64);
    CASE(32);
    CASE(16);
    CASE(8);
    CASE(4);
    CASE(2);
    CASE(1);
#undef CASE
  }
  return ret;
}

Tensor CUDADevice::broadcast_fw_impl(
    const Tensor &x, unsigned dim, unsigned size, Shape &&new_shape) {
  const unsigned skip1 = new_shape.lower_volume(dim);
  const unsigned skip2 = skip1 * size;
  const unsigned total = new_shape.size();
  const unsigned g1 = GRID_SIZE(total, dim1_x_);
  Tensor ret = new_tensor(new_shape);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::broadcast_fw_dev<<<g1, dim1_x_>>>(DATA(ret), CDATA(x), skip1, skip2, total);
  return ret;
}

Tensor CUDADevice::batch_sum_fw_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape().resize_batch(1));
  const unsigned size = ret.shape().size();
  const unsigned g1 = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::batch_sum_fw_dev<<<g1, dim1_x_>>>(
      DATA(ret), CDATA(x), size, x.shape().batch());
  return ret;
}

void CUDADevice::add_gradient_impl(Tensor &a, const Tensor &b) {
  const unsigned nx = a.shape().size();
  const unsigned ny = b.shape().size();
  const unsigned g1 = GRID_SIZE(std::max(nx, ny), dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::add_grad_dev<<<g1, dim1_x_>>>(DATA(a), CDATA(b), nx, ny);
}

void CUDADevice::add_gradient_offset_impl(
    Tensor &a, const Tensor &b, unsigned dim, unsigned offset) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const unsigned base = sa.lower_volume(dim);
  unsigned repeat = 1;
  for (unsigned i = dim + 1; i < sa.depth(); ++i) repeat *= sa[i];
  const unsigned ox = base * offset;
  const unsigned wx = base * sa[dim];
  const unsigned wy = base * sb[dim];
  const unsigned nx = repeat * sa.batch();
  const unsigned ny = repeat * sb.batch();
  const unsigned g1 = GRID_SIZE(wy * std::max(nx, ny), dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::add_grad_ofs_dev<<<g1, dim1_x_>>>(DATA(a) + ox, CDATA(b), wx, wy, nx, ny);
}

void CUDADevice::add_gradient_sparse_impl(
    Tensor &a, const Tensor &b,
    unsigned dim, const std::vector<unsigned>& ids) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const unsigned size = sb.volume();
  const unsigned base = sb.lower_volume(dim);
  const unsigned repeat = size / base;
  const unsigned wx = base * sa[dim];
  const unsigned g1 = GRID_SIZE(size, dim1_x_);
  const unsigned bs = sb.batch();
  const unsigned skip_a = (sa.has_batch()) * sa.volume();
  const unsigned skip_i = ids.size() > 1;
  float *dest = DATA(a);
  const float *src = CDATA(b);

  CUDA_CALL(::cudaSetDevice(dev_id_));
  for (unsigned batch = 0; batch < bs; ++batch) {
    ::add_grad_sparse_dev<<<g1, dim1_x_>>>(
        dest + batch * skip_a + base * ids[batch * skip_i],
        src + batch * size,
        wx, base, repeat);
  }
}

}  // namespace primitiv
