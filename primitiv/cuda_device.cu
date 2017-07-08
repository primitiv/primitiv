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
#define IDY (threadIdx.y + blockIdx.y * blockDim.y)

__global__ void set_const_dev(float k, unsigned size, float *py) {
  const unsigned i = IDX;
  if (i < size) py[i] = k;
}

__global__ void rand_bernoulli_dev(float p, float size, float *py) {
  const unsigned i = IDX;
  if (i < size) py[i] = (float)(py[i] <= p);
}

__global__ void rand_affine_dev(
    float shift, float scale, unsigned size, float *py) {
  const unsigned i = IDX;
  if (i < size) py[i] = py[i] * scale + shift;
}

__global__ void pick_fw_dev(
    const float *px, const unsigned *pi,
    unsigned wx, unsigned wy, unsigned sx, unsigned si, unsigned sy,
    float *py) {
  const unsigned t = IDX;
  const unsigned ox = blockIdx.y * sx + pi[blockIdx.y * si] * wy;
  const unsigned oy = blockIdx.y * sy;
  if (t < sy) py[oy + t] = px[ox + (t / wy) * wx + (t % wy)];
}

__global__ void slice_fw_dev(
    const float *px, unsigned span, unsigned skip, unsigned size, float *py) {
  const unsigned i = IDX;
  if (i < size) py[i] = px[(i / span) * skip + (i % span)];
}

__global__ void concat_fw_dev(
    const float *px, unsigned span, unsigned skip, unsigned x_size,
    unsigned y_size, float *py) {
  const unsigned i = IDX;
  if (i < y_size) py[(i / span) * skip + (i % span)] = px[i % x_size];
}

__global__ void pick_bw_dev(
    const float *pgy, const unsigned *pi,
    unsigned wx, unsigned wy, unsigned sx, unsigned si, unsigned sy,
    float *pgx) {
  const unsigned t = IDX;
  const unsigned ox = blockIdx.y * sx + pi[blockIdx.y * si] * wy;
  const unsigned oy = blockIdx.y * sy;
  if (t < sy) ::atomicAdd(pgx + ox + (t / wy) * wx + (t % wy), pgy[oy + t]);
}

__global__ void slice_bw_dev(
    const float *pgy, unsigned wx, unsigned wy, unsigned nx, unsigned ny,
    float *pgx) {
  const unsigned i = IDX;
  if (i < wy * ::max(nx, ny)) {
    ::atomicAdd(
        pgx + ((i / wy) * wx + (i % wy)) % (wx * nx), pgy[i % (wy * ny)]);
  }
}

#define CUDADEV_KERNEL_FW_X(name, op) \
__global__ void name##_fw_dev(const float *px, unsigned size, float *py) { \
  const unsigned i = IDX; \
  if (i < size) py[i] = (op); \
}

#define CUDADEV_KERNEL_BW_X(name, op) \
__global__ void name##_bw_dev( \
    const float *px, const float *py, const float *pgy, unsigned size, \
    float *pgx) { \
  static_cast<void>(px); \
  static_cast<void>(py); \
  const unsigned i = IDX; \
  if (i < size) pgx[i] += (op); \
}

#define CUDADEV_KERNEL_FW_X_CONST(name, op) \
__global__ void name##_fw_dev( \
    const float *px, float k, unsigned size, float *py) { \
  const unsigned i = IDX; \
  if (i < size) py[i] = (op); \
}

#define CUDADEV_KERNEL_BW_X_CONST(name, op) \
__global__ void name##_bw_dev( \
    const float *px, const float *py, const float *pgy, float k, \
    unsigned size, float *pgx) { \
  static_cast<void>(px); \
  static_cast<void>(py); \
  const unsigned i = IDX; \
  if (i < size) pgx[i] += (op); \
}

#define CUDADEV_KERNEL_FW_X_SCALAR_R(name, op) \
__global__ void name##_fw_dev( \
    const float *px, const float *pk, unsigned size, unsigned mbx, \
    unsigned mbk, float *py) { \
  const unsigned i = IDX; \
  const unsigned shift = blockIdx.y * size; \
  if (i < size) py[i + shift] = op(px[i + mbx * shift], pk[mbk * blockIdx.y]); \
}

#define CUDADEV_KERNEL_FW_X_SCALAR_L(name, op) \
__global__ void name##_fw_dev( \
    const float *px, const float *pk, unsigned size, unsigned mbx, \
    unsigned mbk, float *py) { \
  const unsigned i = IDX; \
  const unsigned shift = blockIdx.y * size; \
  if (i < size) py[i + shift] = op(pk[mbk * blockIdx.y], px[i + mbx * shift]); \
}

#define CUDADEV_KERNEL_FW_AB(name, op) \
__global__ void name##_fw_dev( \
    const float *pa, const float *pb, unsigned size, unsigned mba, \
    unsigned mbb, float *py) { \
  const unsigned i = IDX; \
  const unsigned shift = blockIdx.y * size; \
  if (i < size) py[i + shift] = op(pa[i + mba * shift], pb[i + mbb * shift]); \
}

CUDADEV_KERNEL_FW_X(negate, -px[i]);
CUDADEV_KERNEL_FW_X(sqrt, ::__fsqrt_rn(px[i]));
CUDADEV_KERNEL_FW_X(exp, ::expf(px[i]));
CUDADEV_KERNEL_FW_X(tanh, ::tanhf(px[i]));
CUDADEV_KERNEL_FW_X(sigmoid, .5f + .5f * ::tanhf(.5f * px[i]));
CUDADEV_KERNEL_FW_X(
    softplus, ::fmaxf(px[i], .0f) + ::logf(1.f + ::expf(-::fabs(px[i]))));
CUDADEV_KERNEL_FW_X(sin, ::sinf(px[i]));
CUDADEV_KERNEL_FW_X(cos, ::cosf(px[i]));
CUDADEV_KERNEL_FW_X(tan, ::tanf(px[i]));

CUDADEV_KERNEL_BW_X(negate, -pgy[i]);
CUDADEV_KERNEL_BW_X(sqrt, .5f * pgy[i] / py[i]);
CUDADEV_KERNEL_BW_X(exp, py[i] * pgy[i]);
CUDADEV_KERNEL_BW_X(tanh, (1.f - py[i] * py[i]) * pgy[i]);
CUDADEV_KERNEL_BW_X(sigmoid, py[i] * (1.f - py[i]) * pgy[i]);
CUDADEV_KERNEL_BW_X(softplus, (.5f + .5f * ::tanhf(.5f * px[i])) * pgy[i]);
CUDADEV_KERNEL_BW_X(sin, ::cosf(px[i]) * pgy[i]);
CUDADEV_KERNEL_BW_X(cos, -::sinf(px[i]) * pgy[i]);
CUDADEV_KERNEL_BW_X(tan, (1.f + py[i] * py[i]) * pgy[i]);

CUDADEV_KERNEL_FW_X_CONST(add_const, px[i] + k);
CUDADEV_KERNEL_FW_X_CONST(subtract_const_r, px[i] - k);
CUDADEV_KERNEL_FW_X_CONST(subtract_const_l, k - px[i]);
CUDADEV_KERNEL_FW_X_CONST(multiply_const, px[i] * k);
CUDADEV_KERNEL_FW_X_CONST(divide_const_r, px[i] / k);
CUDADEV_KERNEL_FW_X_CONST(divide_const_l, k / px[i]);
CUDADEV_KERNEL_FW_X_CONST(prelu, ::fmaxf(px[i], .0f) + k * ::fminf(px[i], .0f));
CUDADEV_KERNEL_FW_X_CONST(
    elu, ::fmaxf(px[i], .0f) + k * (::expf(::fminf(px[i], .0f)) - 1.0f));

CUDADEV_KERNEL_BW_X_CONST(add_const, pgy[i]);
CUDADEV_KERNEL_BW_X_CONST(subtract_const_r, pgy[i]);
CUDADEV_KERNEL_BW_X_CONST(subtract_const_l, -pgy[i]);
CUDADEV_KERNEL_BW_X_CONST(multiply_const, k * pgy[i]);
CUDADEV_KERNEL_BW_X_CONST(divide_const_r, pgy[i] / k);
CUDADEV_KERNEL_BW_X_CONST(divide_const_l, -py[i] * pgy[i] / px[i]);
CUDADEV_KERNEL_BW_X_CONST(prelu, pgy[i] * ((px[i] > .0f) + k * (px[i] <= .0f)));
CUDADEV_KERNEL_BW_X_CONST(
    elu, pgy[i] * ((px[i] > .0f) + (py[i] + k) * (px[i] <= .0f)));

CUDADEV_KERNEL_FW_X_SCALAR_R(add_scalar, ::__fadd_rn);
CUDADEV_KERNEL_FW_X_SCALAR_R(subtract_scalar_r, ::__fsub_rn);
CUDADEV_KERNEL_FW_X_SCALAR_L(subtract_scalar_l, ::__fsub_rn);
CUDADEV_KERNEL_FW_X_SCALAR_R(multiply_scalar, ::__fmul_rn);
CUDADEV_KERNEL_FW_X_SCALAR_R(divide_scalar_r, ::__fdiv_rn);
CUDADEV_KERNEL_FW_X_SCALAR_L(divide_scalar_l, ::__fdiv_rn);

CUDADEV_KERNEL_FW_AB(add, ::__fadd_rn);
CUDADEV_KERNEL_FW_AB(subtract, ::__fsub_rn);
CUDADEV_KERNEL_FW_AB(multiply, ::__fmul_rn);
CUDADEV_KERNEL_FW_AB(divide, ::__fdiv_rn);

#undef CUDADEV_KERNEL_FW_X
#undef CUDADEV_KERNEL_BW_X
#undef CUDADEV_KERNEL_FW_X_CONST
#undef CUDADEV_KERNEL_BW_X_CONST
#undef CUDADEV_KERNEL_FW_X_SCALAR_R
#undef CUDADEV_KERNEL_FW_X_SCALAR_L
#undef CUDADEV_KERNEL_FW_AB

__global__ void add_bw_dev(
    const float *, const float *, const float *, const float *pgy,
    unsigned size, unsigned mba, unsigned mbb, float *pga, float *pgb) {
  const unsigned i = IDX;
  const unsigned shift = blockIdx.y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    ::atomicAdd(pga + i + mba * shift, gy);
    ::atomicAdd(pgb + i + mbb * shift, gy);
  }
}

__global__ void subtract_bw_dev(
    const float *, const float *, const float *, const float *pgy,
    unsigned size, unsigned mba, unsigned mbb, float *pga, float *pgb) {
  const unsigned i = IDX;
  const unsigned shift = blockIdx.y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    ::atomicAdd(pga + i + mba * shift, gy);
    ::atomicAdd(pgb + i + mbb * shift, -gy);
  }
}

__global__ void multiply_bw_dev(
    const float *pa, const float *pb, const float *, const float *pgy,
    unsigned size, unsigned mba, unsigned mbb, float *pga, float *pgb) {
  const unsigned i = IDX;
  const unsigned shift = blockIdx.y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    const unsigned a_ofs = i + mba * shift;
    const unsigned b_ofs = i + mbb * shift;
    ::atomicAdd(pga + a_ofs, gy * pb[b_ofs]);
    ::atomicAdd(pgb + b_ofs, gy * pa[a_ofs]);
  }
}

__global__ void divide_bw_dev(
    const float *, const float *pb, const float *py, const float *pgy,
    unsigned size, unsigned mba, unsigned mbb, float *pga, float *pgb) {
  const unsigned i = IDX;
  const unsigned shift = blockIdx.y * size;
  if (i < size) {
    const unsigned b_ofs = i + mbb * shift;
    const unsigned y_ofs = i + shift;
    const float k = pgy[y_ofs] / pb[b_ofs];
    ::atomicAdd(pga + i + mba * shift, k);
    ::atomicAdd(pgb + b_ofs, -k * py[y_ofs]);
  }
}

__global__ void transpose_fw_dev(
    const float *px, unsigned rows, unsigned cols, float *py) {
  const unsigned i = IDX;
  const unsigned j = IDY;
  unsigned ofs = blockIdx.z * rows * cols;
  if (i < rows && j < cols) py[ofs + j + i * cols] = px[ofs + i + j * rows];
}

__global__ void transpose_bw_dev(
    const float *py, unsigned rows, unsigned cols, float *px) {
  const unsigned i = IDX;
  const unsigned j = IDY;
  unsigned ofs = blockIdx.z * rows * cols;
  if (i < rows && j < cols) px[ofs + i + j * rows] += py[ofs + j + i * cols];
}

template<unsigned BLOCK_SIZE>
__global__ void sum_fw_dev(
    const float *px, unsigned skip, unsigned n, float *py) {
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
    const float *px, unsigned skip, unsigned n, float *py) {
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
    const float *px, unsigned skip1, unsigned skip2, unsigned size, float *py) {
  const unsigned i = IDX;
  if (i < size) py[i] = px[i % skip1 + (i / skip2) * skip1];
}

__global__ void batch_sum_fw_dev(
    const float *px, unsigned size, unsigned batch, float *py) {
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

__global__ void inplace_add_dev(
    const float *pgy, unsigned nx, unsigned ny, float *pgx) {
  const unsigned i = IDX;
  if (i < ::max(nx, ny)) ::atomicAdd(pgx + i % nx, pgy[i % ny]);
}

#undef IDX
#undef IDY

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
  max_batch_ = prop.maxGridSize[1];

  cerr << "Configurations:" << endl;
  cerr << "  1 dim ........... " << dim1_x_ << " threads" << endl;
  cerr << "  2 dims .......... " << dim2_x_ << "x"
                                 << dim2_y_ << " threads" << endl;
  cerr << "  Maximum batch ... " << max_batch_ <<endl;

  // Initializes additional libraries
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUBLAS_CALL(::cublasCreate(&cublas_));
  CURAND_CALL(::curandCreateGenerator(&curand_, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(::curandSetPseudoRandomGeneratorSeed(curand_, rng_seed_));

  // Initializes the device pointer for integer IDs.
  ids_ptr_ = pool_.allocate(sizeof(unsigned) * max_batch_);
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

void CUDADevice::reset_tensor_impl(float k, Tensor &x) {
  const unsigned size = x.shape().size();
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::set_const_dev<<<num_blocks, dim1_x_>>>(k, size, DATA(x));
}

void CUDADevice::reset_tensor_by_array_impl(const float values[], Tensor &x) {
  const unsigned size = x.shape().size();
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        x.data(), values, sizeof(float) * size, cudaMemcpyHostToDevice));
}

void CUDADevice::copy_tensor_impl(const Tensor &x, Tensor &y) {
  switch (x.device()->type()) {
    case Device::DEVICE_TYPE_CPU:
      reset_tensor_by_array(CDATA(x), y);
      break;
    case Device::DEVICE_TYPE_CUDA:
      CUDA_CALL(::cudaSetDevice(dev_id_));
      // NOTE(odashi):
      // If source/destination devices use the unified memory space on the 64
      // bits machine, we can perform ::cudaMemcpy to copy data beyond devices.
      CUDA_CALL(::cudaMemcpyAsync(
            DATA(y), CDATA(x),
            sizeof(float) * x.shape().size(),
            cudaMemcpyDeviceToDevice, 0));
      break;
    default:
      reset_tensor_by_vector(x.to_vector(), y);
  }
}

void CUDADevice::random_bernoulli_impl(float p, Tensor &y) {
  const unsigned size = y.shape().size();
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateUniform(curand_, DATA(y), size));
  ::rand_bernoulli_dev<<<num_blocks, dim1_x_>>>(p, size, DATA(y));
}

void CUDADevice::random_uniform_impl(float lower, float upper, Tensor &y) {
  const unsigned size = y.shape().size();
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_);
  const float scale = upper - lower;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateUniform(curand_, DATA(y), size));
  ::rand_affine_dev<<<num_blocks, dim1_x_>>>(lower, scale, size, DATA(y));
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

void CUDADevice::pick_fw_impl(
    const Tensor &x, unsigned dim, const std::vector<unsigned> &ids,
    Tensor &y) {
  const unsigned wy = y.shape().lower_volume(dim);
  const unsigned sy = y.shape().volume();
  const unsigned g1 = GRID_SIZE(sy, dim1_x_);
  const unsigned bs = y.shape().batch();

  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        ids_ptr_.get(), ids.data(), sizeof(unsigned) * ids.size(),
        cudaMemcpyHostToDevice));
  ::pick_fw_dev<<<dim3(g1, bs), dim1_x_>>>(
      CDATA(x), static_cast<const unsigned *>(ids_ptr_.get()),
      wy * x.shape()[dim], wy,
      x.shape().has_batch() * x.shape().volume(), ids.size() > 1, sy,
      DATA(y));
}

void CUDADevice::slice_fw_impl(
    const Tensor &x, unsigned dim, unsigned offset, Tensor &y) {
  const unsigned base = y.shape().lower_volume(dim);
  const unsigned span = base * y.shape()[dim];
  const unsigned skip = base * x.shape()[dim];
  const unsigned size = y.shape().size();
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::slice_fw_dev<<<num_blocks, dim1_x_>>>(
      CDATA(x) + base * offset, span, skip, size, DATA(y));
}

void CUDADevice::concat_fw_impl(
    const std::vector<const Tensor *> &xs, unsigned dim, Tensor &y) {
  const unsigned new_bs = y.shape().batch();
  const unsigned base = y.shape().lower_volume(dim);
  const unsigned skip = base * y.shape()[dim];
  unsigned repeat = y.shape().volume() / skip;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  unsigned offset = 0;
  for (const Tensor *x : xs) {
    const unsigned span = base * x->shape()[dim];
    const unsigned x_size = span * repeat * x->shape().batch();
    const unsigned y_size = span * repeat * new_bs;
    const unsigned num_blocks = GRID_SIZE(y_size, dim1_x_);
    ::concat_fw_dev<<<num_blocks, dim1_x_>>>(
       CDATA(*x), span, skip, x_size, y_size, DATA(y) + offset);
    offset += span;
  }
}

void CUDADevice::pick_bw_impl(
    const Tensor &gy, unsigned dim, const std::vector<unsigned>& ids,
    Tensor &gx) {
  const unsigned wy = gy.shape().lower_volume(dim);
  const unsigned sy = gy.shape().volume();
  const unsigned g1 = GRID_SIZE(sy, dim1_x_);
  const unsigned bs = gy.shape().batch();

  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        ids_ptr_.get(), ids.data(), sizeof(unsigned) * ids.size(),
        cudaMemcpyHostToDevice));
  ::pick_bw_dev<<<dim3(g1, bs), dim1_x_>>>(
      CDATA(gy), static_cast<const unsigned *>(ids_ptr_.get()),
      wy *gx.shape()[dim], wy,
      gx.shape().has_batch() * gx.shape().volume(), ids.size() > 1, sy,
      DATA(gx));
}

void CUDADevice::slice_bw_impl(
    const Tensor &gy, unsigned dim, unsigned offset, Tensor &gx) {
  const Shape &sx = gx.shape();
  const Shape &sy = gy.shape();
  const unsigned base = sx.lower_volume(dim);
  const unsigned ox = base * offset;
  const unsigned wx = base * sx[dim];
  const unsigned wy = base * sy[dim];
  const unsigned repeat = sx.volume() / wx;
  const unsigned nx = repeat * sx.batch();
  const unsigned ny = repeat * sy.batch();
  const unsigned g1 = GRID_SIZE(wy * std::max(nx, ny), dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::slice_bw_dev<<<g1, dim1_x_>>>(CDATA(gy), wx, wy, nx, ny, DATA(gx) + ox);
}

#define CUDADEV_FW_X(name) \
void CUDADevice::name##_fw_impl(const Tensor &x, Tensor &y) { \
  const unsigned size = x.shape().size(); \
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<num_blocks, dim1_x_>>>(CDATA(x), size, DATA(y)); \
}

#define CUDADEV_BW_X(name) \
void CUDADevice::name##_bw_impl( \
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) { \
  const unsigned size = x.shape().size(); \
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_bw_dev<<<num_blocks, dim1_x_>>>( \
      CDATA(x), CDATA(y), CDATA(gy), size, DATA(gx)); \
}

#define CUDADEV_FW_X_CONST(name) \
void CUDADevice::name##_fw_impl(const Tensor &x, float k, Tensor &y) { \
  const unsigned size = x.shape().size(); \
  const unsigned num_blocks = GRID_SIZE(size,dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<num_blocks, dim1_x_>>>(CDATA(x), k, size, DATA(y)); \
}

#define CUDADEV_BW_X_CONST(name) \
void CUDADevice::name##_bw_impl( \
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) { \
  const unsigned size = x.shape().size(); \
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_bw_dev<<<num_blocks, dim1_x_>>>( \
      CDATA(x), CDATA(y), CDATA(gy), k, size, DATA(gx)); \
}

#define CUDADEV_FW_X_SCALAR(name) \
void CUDADevice::name##_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) { \
  const unsigned size = y.shape().volume(); \
  const unsigned g1 = GRID_SIZE(size, dim1_x_); \
  const unsigned g2 = y.shape().batch(); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<dim3(g1, g2, 1), dim1_x_>>>( \
      CDATA(x), CDATA(k), size, \
      x.shape().has_batch(), k.shape().has_batch(), DATA(y)); \
}

#define CUDADEV_FW_AB(name) \
void CUDADevice::name##_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) { \
  const unsigned size = y.shape().volume(); \
  const unsigned g1 = GRID_SIZE(size, dim1_x_); \
  const unsigned g2 = y.shape().batch(); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<dim3(g1, g2, 1), dim1_x_>>>( \
      CDATA(a), CDATA(b), size, \
      a.shape().has_batch(), b.shape().has_batch(), DATA(y)); \
}

#define CUDADEV_BW_AB(name) \
void CUDADevice::name##_bw_impl( \
    const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, \
    Tensor &ga, Tensor &gb) { \
  const unsigned size = y.shape().volume(); \
  const unsigned g1 = GRID_SIZE(size, dim1_x_); \
  const unsigned g2 = y.shape().batch(); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_bw_dev<<<dim3(g1, g2, 1), dim1_x_>>>( \
      CDATA(a), CDATA(b), CDATA(y), CDATA(gy), size, \
      a.shape().has_batch(), b.shape().has_batch(), DATA(ga), DATA(gb)); \
}

CUDADEV_FW_X(negate);
CUDADEV_FW_X(sqrt);
CUDADEV_FW_X(exp);
CUDADEV_FW_X(tanh);
CUDADEV_FW_X(sigmoid);
CUDADEV_FW_X(softplus);
CUDADEV_FW_X(sin);
CUDADEV_FW_X(cos);
CUDADEV_FW_X(tan);

CUDADEV_BW_X(negate);
CUDADEV_BW_X(sqrt);
CUDADEV_BW_X(exp);
CUDADEV_BW_X(tanh);
CUDADEV_BW_X(sigmoid);
CUDADEV_BW_X(softplus);
CUDADEV_BW_X(sin);
CUDADEV_BW_X(cos);
CUDADEV_BW_X(tan);

CUDADEV_FW_X_CONST(add_const);
CUDADEV_FW_X_CONST(subtract_const_r);
CUDADEV_FW_X_CONST(subtract_const_l);
CUDADEV_FW_X_CONST(multiply_const);
CUDADEV_FW_X_CONST(divide_const_r);
CUDADEV_FW_X_CONST(divide_const_l);
CUDADEV_FW_X_CONST(prelu);
CUDADEV_FW_X_CONST(elu);

CUDADEV_BW_X_CONST(add_const);
CUDADEV_BW_X_CONST(subtract_const_r);
CUDADEV_BW_X_CONST(subtract_const_l);
CUDADEV_BW_X_CONST(multiply_const);
CUDADEV_BW_X_CONST(divide_const_r);
CUDADEV_BW_X_CONST(divide_const_l);
CUDADEV_BW_X_CONST(prelu);
CUDADEV_BW_X_CONST(elu);

CUDADEV_FW_X_SCALAR(add_scalar);
CUDADEV_FW_X_SCALAR(subtract_scalar_r);
CUDADEV_FW_X_SCALAR(subtract_scalar_l);
CUDADEV_FW_X_SCALAR(multiply_scalar);
CUDADEV_FW_X_SCALAR(divide_scalar_r);
CUDADEV_FW_X_SCALAR(divide_scalar_l);

CUDADEV_FW_AB(add);
CUDADEV_FW_AB(subtract);
CUDADEV_FW_AB(multiply);
CUDADEV_FW_AB(divide);

CUDADEV_BW_AB(add);
CUDADEV_BW_AB(subtract);
CUDADEV_BW_AB(multiply);
CUDADEV_BW_AB(divide);

#undef CUDADEV_FW_X
#undef CUDADEV_BW_X
#undef CUDADEV_FW_X_CONST
#undef CUDADEV_BW_X_CONST
#undef CUDADEV_FW_X_SCALAR
#undef CUDADEV_FW_AB
#undef CUDADEV_BW_AB

void CUDADevice::transpose_fw_impl(const Tensor &x, Tensor &y) {
  const unsigned rows = x.shape()[0];
  const unsigned cols = x.shape()[1];
  const unsigned bs = x.shape().batch();
  const unsigned g1 = GRID_SIZE(rows, dim2_x_);
  const unsigned g2 = GRID_SIZE(cols, dim2_y_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::transpose_fw_dev<<<dim3(g1, g2, bs), dim3(dim2_x_, dim2_y_, 1)>>>(
      CDATA(x), rows, cols, DATA(y));
}

void CUDADevice::matmul_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
  const unsigned di = a.shape()[0];
  const unsigned dj = a.shape()[1];
  const unsigned dk = b.shape()[1];
  float alpha = 1.;
  float beta = 0.;
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
            &beta, DATA(y) + n * y_skip, di));
    }
  } else {
    // Do gemm only once to calculate the product with a combined matrix.
    CUBLAS_CALL(::cublasSgemm(
          cublas_, ::CUBLAS_OP_N, ::CUBLAS_OP_N,
          di, dk * b.shape().batch(), dj,
          &alpha, CDATA(a), di, CDATA(b), dj,
          &beta, DATA(y), di));
  }
}

void CUDADevice::transpose_bw_impl(
    const Tensor &, const Tensor &, const Tensor &gy, Tensor &gx) {
  const unsigned rows = gx.shape()[0];
  const unsigned cols = gx.shape()[1];
  const unsigned bs = gx.shape().batch();
  const unsigned g1 = GRID_SIZE(rows, dim2_x_);
  const unsigned g2 = GRID_SIZE(cols, dim2_y_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::transpose_bw_dev<<<dim3(g1, g2, bs), dim3(dim2_x_, dim2_y_, 1)>>>(
      CDATA(gy), rows, cols, DATA(gx));
}

void CUDADevice::matmul_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &, const Tensor &gy,
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

void CUDADevice::sum_fw_impl(const Tensor &x, unsigned dim, Tensor &y) {
  const unsigned n = x.shape()[dim];
  const unsigned r = y.shape().size();
  const unsigned s = y.shape().lower_volume(dim);
  unsigned block_size = dim1_x_;
  while (block_size >> 1 >= n) block_size >>= 1;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  switch (block_size) {
#define CASE(k) \
    case k: ::sum_fw_dev<k><<<r, k>>>(CDATA(x), s, n, DATA(y)); break
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
}

void CUDADevice::logsumexp_fw_impl(const Tensor &x, unsigned dim, Tensor &y) {
  const unsigned n = x.shape()[dim];
  const unsigned r = y.shape().size();
  const unsigned s = y.shape().lower_volume(dim);
  unsigned block_size = dim1_x_;
  while (block_size >> 1 >= n) block_size >>= 1;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  switch (block_size) {
#define CASE(k) \
    case k: ::logsumexp_fw_dev<k><<<r, k>>>(CDATA(x), s, n, DATA(y)); break
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
}

void CUDADevice::broadcast_fw_impl(
    const Tensor &x, unsigned dim, unsigned size, Tensor &y) {
  const unsigned skip1 = y.shape().lower_volume(dim);
  const unsigned skip2 = skip1 * size;
  const unsigned total = y.shape().size();
  const unsigned g1 = GRID_SIZE(total, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::broadcast_fw_dev<<<g1, dim1_x_>>>(CDATA(x), skip1, skip2, total, DATA(y));
}

void CUDADevice::batch_sum_fw_impl(const Tensor &x, Tensor &y) {
  const unsigned size = y.shape().size();
  const unsigned g1 = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::batch_sum_fw_dev<<<g1, dim1_x_>>>(
      CDATA(x), size, x.shape().batch(), DATA(y));
}

void CUDADevice::inplace_add_impl(const Tensor &gy, Tensor &gx) {
  const unsigned nx = gx.shape().size();
  const unsigned ny = gy.shape().size();
  const unsigned g1 = GRID_SIZE(std::max(nx, ny), dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::inplace_add_dev<<<g1, dim1_x_>>>(CDATA(gy), nx, ny, DATA(gx));
}

}  // namespace primitiv
