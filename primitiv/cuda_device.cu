#include <config.h>

#include <iostream>
#include <random>
#include <primitiv/cuda_device.h>
#include <primitiv/error.h>

using std::cerr;
using std::endl;

namespace {

/**
 * Retrieves cuBLAS error string.
 * @param err cuBLAS error cude.
 * @return Error string.
 */
std::string cublasGetErrorString(::cublasStatus_t err) {
#define MESSAGE(status) if (err == status) return #status
  MESSAGE(CUBLAS_STATUS_SUCCESS);
  MESSAGE(CUBLAS_STATUS_NOT_INITIALIZED);
  MESSAGE(CUBLAS_STATUS_ALLOC_FAILED);
  MESSAGE(CUBLAS_STATUS_INVALID_VALUE);
  MESSAGE(CUBLAS_STATUS_ARCH_MISMATCH);
  MESSAGE(CUBLAS_STATUS_MAPPING_ERROR);
  MESSAGE(CUBLAS_STATUS_EXECUTION_FAILED);
  MESSAGE(CUBLAS_STATUS_INTERNAL_ERROR);
  MESSAGE(CUBLAS_STATUS_NOT_SUPPORTED);
  MESSAGE(CUBLAS_STATUS_LICENSE_ERROR);
#undef MESSAGE
  return "Unknown cublasStatus_t value.";
}

/**
 * Retrieves cuRAND error string.
 * @param err cuRAND error cude.
 * @return Error string.
 */
std::string curandGetErrorString(::curandStatus_t err) {
#define MESSAGE(status) if (err == status) return #status
  MESSAGE(CURAND_STATUS_SUCCESS);
  MESSAGE(CURAND_STATUS_VERSION_MISMATCH);
  MESSAGE(CURAND_STATUS_NOT_INITIALIZED);
  MESSAGE(CURAND_STATUS_ALLOCATION_FAILED);
  MESSAGE(CURAND_STATUS_TYPE_ERROR);
  MESSAGE(CURAND_STATUS_OUT_OF_RANGE);
  MESSAGE(CURAND_STATUS_LENGTH_NOT_MULTIPLE);
  MESSAGE(CURAND_STATUS_DOUBLE_PRECISION_REQUIRED);
  MESSAGE(CURAND_STATUS_LAUNCH_FAILURE);
  MESSAGE(CURAND_STATUS_PREEXISTING_FAILURE);
  MESSAGE(CURAND_STATUS_INITIALIZATION_FAILED);
  MESSAGE(CURAND_STATUS_ARCH_MISMATCH);
  MESSAGE(CURAND_STATUS_INTERNAL_ERROR);
#undef MESSAGE
  return "Unknown curandStatus_t value.";
}

}

#define CUDA_CALL(f) { \
  ::cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    THROW_ERROR( \
        "CUDA function failed. statement: " << #f \
        << ", error: " << err \
        << ": " << ::cudaGetErrorString(err)); \
  } \
}

#define CUBLAS_CALL(f) { \
  ::cublasStatus_t err = (f); \
  if (err != CUBLAS_STATUS_SUCCESS) { \
    THROW_ERROR( \
        "CUBLAS function failed. statement: " << #f \
        << ", error: " << err \
        << ": " << ::cublasGetErrorString(err)); \
  } \
}

#define CURAND_CALL(f) { \
  ::curandStatus_t err = (f); \
  if (err != CURAND_STATUS_SUCCESS) { \
    THROW_ERROR( \
        "CURAND function failed. statement: " << #f \
        << ", error: " << err \
        << ": " << ::curandGetErrorString(err)); \
  } \
}

namespace {

/*
 * CUDA kernels
 */

#define IDX (threadIdx.x + blockIdx.x * blockDim.x)

__global__ void dev_set_const(float *py, float k, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = k;
}

__global__ void dev_rand_bernoulli(float *px, float p, float size) {
  const unsigned i = IDX;
  if (i < size) px[i] = (float)(px[i] <= p);
}

__global__ void dev_rand_affine(
    float *px, float shift, float scale, unsigned size) {
  const unsigned i = IDX;
  if (i < size) px[i] = px[i] * scale + shift;
}

__global__ void dev_slice(
    float *py, const float *px, unsigned span, unsigned skip, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = px[(i / span) * skip + (i % span)];
}

__global__ void dev_concat(
    float *py, const float *px,
    unsigned span, unsigned skip, unsigned x_size, unsigned y_size) {
  const unsigned i = IDX;
  if (i < y_size) py[(i / span) * skip + (i % span)] = px[i % x_size];
}

__global__ void dev_negate(float *py, const float *px, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = -px[i];
}

__global__ void dev_add_const(
    float *py, const float *px, float k, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = px[i] + k;
}

__global__ void dev_subtract_const_l(
    float *py, const float *px, float k, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = k - px[i];
}

__global__ void dev_subtract_const_r(
    float *py, const float *px, float k, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = px[i] - k;
}

__global__ void dev_multiply_const(
    float *py, const float *px, float k, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = px[i] * k;
}

__global__ void dev_divide_const_l(
    float *py, const float *px, float k, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = k / px[i];
}

__global__ void dev_divide_const_r(
    float *py, const float *px, float k, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = px[i] / k;
}

__global__ void dev_add(
    float *py, const float *pa, const float *pb,
    unsigned size, unsigned mba, unsigned mbb) {
  const unsigned i = IDX;
  const unsigned shift = blockIdx.y * size;
  if (i < size) py[i + shift] = pa[i + mba * shift] + pb[i + mbb * shift];
}

__global__ void dev_subtract(
    float *py, const float *pa, const float *pb,
    unsigned size, unsigned mba, unsigned mbb) {
  const unsigned i = IDX;
  const unsigned shift = blockIdx.y * size;
  if (i < size) py[i + shift] = pa[i + mba * shift] - pb[i + mbb * shift];
}

__global__ void dev_multiply(
    float *py, const float *pa, const float *pb,
    unsigned size, unsigned mba, unsigned mbb) {
  const unsigned i = IDX;
  const unsigned shift = blockIdx.y * size;
  if (i < size) py[i + shift] = pa[i + mba * shift] * pb[i + mbb * shift];
}

__global__ void dev_divide(
    float *py, const float *pa, const float *pb,
    unsigned size, unsigned mba, unsigned mbb) {
  const unsigned i = IDX;
  const unsigned shift = blockIdx.y * size;
  if (i < size) py[i + shift] = pa[i + mba * shift] / pb[i + mbb * shift];
}

__global__ void dev_transpose(
    float *py, const float *px, unsigned rows, unsigned cols) {
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned j = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned ofs = blockIdx.z * rows * cols;
  if (i < rows && j < cols) {
    py[ofs + j + i * cols] = px[ofs + i + j * rows];
  }
}

__global__ void dev_exp(float *py, const float *px, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = ::expf(px[i]);
}

__global__ void dev_tanh(float *py, const float *px, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = ::tanhf(px[i]);
}

__global__ void dev_sigmoid(float *py, const float *px, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = .5f + .5f * ::tanhf(.5f * px[i]);
}

__global__ void dev_step(float *py, const float *px, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = (float)(px[i] > .0f);
}

__global__ void dev_relu(float *py, const float *px, unsigned size) {
  const unsigned i = IDX;
  if (i < size) py[i] = ::fmaxf(px[i], .0f);
}

__global__ void dev_batch_sum(
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

__global__ void dev_add_grad(
    float *pgx, const float *pgy, unsigned nx, unsigned ny) {
  const unsigned i = IDX;
  if (i < ::max(nx, ny)) ::atomicAdd(pgx + i % nx, pgy[i % ny]);
}

__global__ void dev_add_grad_ofs(
    float *pgx, const float *pgy,
    unsigned wx, unsigned wy, unsigned nx, unsigned ny) {
  const unsigned i = IDX;
  if (i < wy * ::max(nx, ny)) {
    ::atomicAdd(
        pgx + ((i / wy) * wx + (i % wy)) % (wx * nx),
        pgy[i % (wy * ny)]);
  }
}

#undef IDX

}  // namespace

namespace primitiv {

void CUDADevice::initialize() {
  // Retrieves device properties.
  int max_devs;
  CUDA_CALL(::cudaGetDeviceCount(&max_devs));
  if (dev_id_ >= static_cast<unsigned>(max_devs)) {
    THROW_ERROR(
        "Invalid CUDA device ID. given: " << dev_id_ << " >= " << max_devs);
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

  // Additional libraries
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUBLAS_CALL(::cublasCreate(&cublas_));
  CURAND_CALL(::curandCreateGenerator(&curand_, CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CALL(::curandSetPseudoRandomGeneratorSeed(curand_, rng_seed_));
}

CUDADevice::CUDADevice(unsigned device_id)
: dev_id_(device_id)
, rng_seed_(std::random_device()()) {
  initialize();
}

CUDADevice::CUDADevice(unsigned device_id, unsigned rng_seed)
: dev_id_(device_id)
, rng_seed_(rng_seed) {
  initialize();
}

CUDADevice::~CUDADevice() {
  // Check memory leak
  if (!blocks_.empty()) {
    cerr << "FATAL ERROR: Detected memory leak on CUDADevice!" << endl;
    cerr << "Leaked blocks (handle: size):" << endl;
    for (const auto &kv : blocks_) {
      cerr << "  " << kv.first << ": " << kv.second << endl;
    }
    std::abort();
  }

  // Additional libraries
  CUBLAS_CALL(::cublasDestroy(cublas_));
  CURAND_CALL(::curandDestroyGenerator(curand_));
}

void *CUDADevice::new_handle(const Shape &shape) {
  const unsigned mem_size = sizeof(float) * shape.size();
  void *data;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMalloc(&data, mem_size));
  blocks_.insert(std::make_pair(data, mem_size));
  return data;
}

void CUDADevice::delete_tensor_impl(Tensor &x) {
  void *data = x.data();
  auto it = blocks_.find(data);
  if (it == blocks_.end()) {
    THROW_ERROR("Attempted to dispose unknown memory block: " << data);
  }
  blocks_.erase(it);
  CUDA_CALL(::cudaFree(data));
}

#define GRID_SIZE(x, threads) (((x) + (threads) - 1) / (threads))
#define DATA(x) static_cast<float *>((x).data())
#define CDATA(x) static_cast<const float *>((x).data())

std::vector<float> CUDADevice::tensor_to_vector_impl(const Tensor &x) {
  const unsigned size = x.shape().size();
  std::vector<float> ret(size);
  CUDA_CALL(::cudaMemcpy(
        &ret[0], x.data(), sizeof(float) * size, cudaMemcpyDeviceToHost));
  return ret;
}

void CUDADevice::reset_tensor_impl(Tensor &x, float k) {
  const unsigned size = x.shape().size();
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_);
  ::dev_set_const<<<num_blocks, dim1_x_>>>(DATA(x), k, size);
}

void CUDADevice::reset_tensor_impl(
    Tensor &x, const std::vector<float> &values) {
  const unsigned size = x.shape().size();
  CUDA_CALL(::cudaMemcpy(
        x.data(), &values[0], sizeof(float) * size, cudaMemcpyHostToDevice));
}

Tensor CUDADevice::random_bernoulli_impl(const Shape &shape, float p) {
  const unsigned size = shape.size();
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_);
  Tensor ret = new_tensor(shape);
  CURAND_CALL(::curandGenerateUniform(curand_, DATA(ret), size));
  ::dev_rand_bernoulli<<<num_blocks, dim1_x_>>>(DATA(ret), p, size);
  return ret;
}

Tensor CUDADevice::random_uniform_impl(
    const Shape &shape, float lower, float upper) {
  const unsigned size = shape.size();
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_);
  const float scale = upper - lower;
  Tensor ret = new_tensor(shape);
  CURAND_CALL(::curandGenerateUniform(curand_, DATA(ret), size));
  ::dev_rand_affine<<<num_blocks, dim1_x_>>>(DATA(ret), lower, scale, size);
  return ret;
}

Tensor CUDADevice::random_normal_impl(
    const Shape &shape, float mean, float sd) {
  const unsigned size = shape.size();
  Tensor ret = new_tensor(shape);
  CURAND_CALL(::curandGenerateNormal(curand_, DATA(ret), size, mean, sd));
  return ret;
}

Tensor CUDADevice::slice_impl(
    const Tensor &x, unsigned dim, unsigned offset, const Shape &new_shape) {
  unsigned base = 1;
  for (unsigned i = 0; i < dim; ++i) base *= new_shape.dim(i);
  const unsigned span = base * new_shape.dim(dim);
  const unsigned skip = base * x.shape().dim(dim);
  const unsigned size = new_shape.size();
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_);
  Tensor ret = new_tensor(new_shape);
  ::dev_slice<<<num_blocks, dim1_x_>>>(
      DATA(ret), CDATA(x) + base * offset, span, skip, size);
  return ret;
}

Tensor CUDADevice::concat_impl(
    const std::vector<const Tensor *> &xs,
    unsigned dim, const Shape &new_shape) {
  const std::vector<unsigned> new_dims = new_shape.dims();
  const unsigned new_bs = new_shape.batch_size();
  unsigned base = 1;
  for (unsigned i = 0; i < dim; ++i) base *= new_dims[i];
  unsigned repeat = 1;
  for (unsigned i = dim + 1; i < new_dims.size(); ++i) repeat *= new_dims[i];

  Tensor ret = new_tensor(new_shape);
  unsigned offset = 0;
  for (const Tensor *x : xs) {
    const unsigned span = base * x->shape().dim(dim);
    const unsigned skip = base * new_dims[dim];
    const unsigned x_size = span * repeat * x->shape().batch_size();
    const unsigned y_size = span * repeat * new_bs;
    const unsigned num_blocks = GRID_SIZE(y_size, dim1_x_);
    ::dev_concat<<<num_blocks, dim1_x_>>>(
        DATA(ret) + offset, CDATA(*x), span, skip, x_size, y_size);
    offset += span;
  }
  return ret;
}

Tensor CUDADevice::duplicate_impl(const Tensor &x) {
  Tensor ret = new_tensor(x.shape());
  CUDA_CALL(::cudaMemcpy(
      ret.data(), x.data(), sizeof(float) * x.shape().size(),
      cudaMemcpyDeviceToDevice));
  return ret;
}

#define CUDA_DEV_UNARY(name, kernel) \
Tensor CUDADevice::name(const Tensor &x) { \
  Tensor ret = new_tensor(x.shape()); \
  const unsigned size = x.shape().size(); \
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_); \
  ::kernel<<<num_blocks, dim1_x_>>>(DATA(ret), CDATA(x), size); \
  return ret; \
}

#define CUDA_DEV_BINARY_KX(name, kernel) \
Tensor CUDADevice::name(float k, const Tensor &x) { \
  Tensor ret = new_tensor(x.shape()); \
  const unsigned size = x.shape().size(); \
  const unsigned num_blocks = GRID_SIZE(size, dim1_x_); \
  ::kernel<<<num_blocks, dim1_x_>>>(DATA(ret), CDATA(x), k, size); \
  return ret; \
}

#define CUDA_DEV_BINARY_XK(name, kernel) \
Tensor CUDADevice::name(const Tensor &x, float k) { \
  Tensor ret = new_tensor(x.shape()); \
  const unsigned size = x.shape().size(); \
  const unsigned num_blocks = GRID_SIZE(size,dim1_x_); \
  ::kernel<<<num_blocks, dim1_x_>>>(DATA(ret), CDATA(x), k, size); \
  return ret; \
}

#define CUDA_DEV_BINARY_AB(name, kernel) \
Tensor CUDADevice::name(const Tensor &a, const Tensor &b) { \
  const Shape &sa = a.shape(); \
  const Shape &sb = b.shape(); \
  const unsigned ba = sa.batch_size(); \
  const unsigned bb = sb.batch_size(); \
  const unsigned size = sa.size() / ba; \
  const unsigned x = GRID_SIZE(size, dim1_x_); \
  const unsigned y = std::max(ba, bb); \
  Tensor ret = new_tensor(Shape(sa.dims(), y)); \
  ::kernel<<<dim3(x, y, 1), dim1_x_>>>( \
      DATA(ret), CDATA(a), CDATA(b), size, ba > 1, bb > 1); \
  return ret; \
}

CUDA_DEV_UNARY(negate_impl, dev_negate);
CUDA_DEV_UNARY(exp_impl, dev_exp);
CUDA_DEV_UNARY(tanh_impl, dev_tanh);
CUDA_DEV_UNARY(sigmoid_impl, dev_sigmoid);
CUDA_DEV_UNARY(step_impl, dev_step);
CUDA_DEV_UNARY(relu_impl, dev_relu);

CUDA_DEV_BINARY_XK(add_impl, dev_add_const);
CUDA_DEV_BINARY_KX(subtract_impl, dev_subtract_const_l);
CUDA_DEV_BINARY_XK(subtract_impl, dev_subtract_const_r);
CUDA_DEV_BINARY_XK(multiply_impl, dev_multiply_const);
CUDA_DEV_BINARY_KX(divide_impl, dev_divide_const_l);
CUDA_DEV_BINARY_XK(divide_impl, dev_divide_const_r);

CUDA_DEV_BINARY_AB(add_impl, dev_add);
CUDA_DEV_BINARY_AB(subtract_impl, dev_subtract);
CUDA_DEV_BINARY_AB(multiply_impl, dev_multiply);
CUDA_DEV_BINARY_AB(divide_impl, dev_divide);

#undef CUDA_DEV_UNARY
#undef CUDA_DEV_BINARY_KX
#undef CUDA_DEV_BINARY_XK
#undef CUDA_DEV_BINARY_AB

Tensor CUDADevice::transpose_impl(const Tensor &x) {
  const Shape &s = x.shape();
  const unsigned d1 = s.dim(0);
  const unsigned d2 = s.dim(1);
  const unsigned bs = s.batch_size();
  const unsigned g1 = GRID_SIZE(d1, dim2_x_);
  const unsigned g2 = GRID_SIZE(d2, dim2_y_);
  Tensor ret = new_tensor(Shape({d2, d1}, bs));
  ::dev_transpose<<<dim3(g1, g2, bs), dim3(dim2_x_, dim2_y_, 1)>>>(
      DATA(ret), CDATA(x), d1, d2);
  return ret;
}

Tensor CUDADevice::dot_impl(const Tensor &a, const Tensor &b) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  const unsigned di = sa.dim(0);
  const unsigned dj = sa.dim(1);
  const unsigned dk = sb.dim(1);
  const unsigned ba = sa.batch_size();
  const unsigned bb = sb.batch_size();
  const unsigned bs = std::max(ba, bb);
  float alpha = 1.;
  float beta = 0.;
  Tensor ret = new_tensor(Shape({di, dk}, bs));
  reset_tensor(ret, 0);
  if (ba == 1) {
    // Do gemm only once to calculate dot with combined matrices.
    CUBLAS_CALL(::cublasSgemm(
          cublas_, ::CUBLAS_OP_N, ::CUBLAS_OP_N,
          di, bb * dk, dj,
          &alpha, CDATA(a), di, CDATA(b), dj,
          &beta, DATA(ret), di));
  } else {
    // Do gemm multiple times.
    const unsigned a_skip = di * dj;
    const unsigned b_skip = static_cast<unsigned>(bb > 1) * dj * dk;
    const unsigned y_skip = di * dk;
    for (unsigned n = 0; n < ba; ++n) {
      CUBLAS_CALL(::cublasSgemm(
            cublas_, ::CUBLAS_OP_N, ::CUBLAS_OP_N,
            di, dk, dj,
            &alpha, CDATA(a) + n * a_skip, di, CDATA(b) + n * b_skip, dj,
            &beta, DATA(ret) + n * y_skip, di));
    }
  }
  return ret;
}


Tensor CUDADevice::batch_sum_impl(const Tensor &x) {
  Tensor ret = new_tensor(Shape(x.shape().dims()));
  const unsigned size = ret.shape().size();
  const unsigned g1 = GRID_SIZE(size, dim1_x_);
  ::dev_batch_sum<<<g1, dim1_x_>>>(
      DATA(ret), CDATA(x), size, x.shape().batch_size());
  return ret;
}

void CUDADevice::add_gradient_impl(Tensor &a, const Tensor &b) {
  const unsigned nx = a.shape().size();
  const unsigned ny = b.shape().size();
  const unsigned g1 = GRID_SIZE(std::max(nx, ny), dim1_x_);
  ::dev_add_grad<<<g1, dim1_x_>>>(DATA(a), CDATA(b), nx, ny);
}

void CUDADevice::add_gradient_offset_impl(
    Tensor &a, const Tensor &b, unsigned dim, unsigned offset) {
  const Shape &sa = a.shape();
  const Shape &sb = b.shape();
  unsigned base = 1;
  for (unsigned i = 0; i < dim; ++i) base *= sa.dim(i);
  unsigned repeat = 1;
  for (unsigned i = dim + 1; i < sa.dims().size(); ++i) repeat *= sa.dim(i);
  const unsigned ox = base * offset;
  const unsigned wx = base * sa.dim(dim);
  const unsigned wy = base * sb.dim(dim);
  const unsigned nx = repeat * sa.batch_size();
  const unsigned ny = repeat * sb.batch_size();
  const unsigned g1 = GRID_SIZE(wy * std::max(nx, ny), dim1_x_);
  ::dev_add_grad_ofs<<<g1, dim1_x_>>>(DATA(a) + ox, CDATA(b), wx, wy, nx, ny);
}

}  // namespace primitiv
