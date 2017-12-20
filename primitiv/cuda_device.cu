#include <primitiv/config.h>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <curand.h>
#include <iostream>
#include <random>
#include <primitiv/cuda_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/error.h>
#include <primitiv/memory_pool.h>

using std::cerr;
using std::endl;

namespace {

/*
 * CUDA kernels
 */

#define IDX (threadIdx.x + blockIdx.x * blockDim.x)
#define IDY (threadIdx.y + blockIdx.y * blockDim.y)

__global__ void set_const_dev(float k, std::uint32_t size, float *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = k;
}

__global__ void set_identity_dev(std::uint32_t size, std::uint32_t skip, float *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = !(i % skip);
}

__global__ void rand_bernoulli_dev(float p, float size, float *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = (float)(py[i] <= p);
}

__global__ void rand_affine_dev(
    float shift, float scale, std::uint32_t size, float *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = py[i] * scale + shift;
}

__global__ void pick_fw_dev(
    const float *px, const std::uint32_t *pi,
    std::uint32_t wx, std::uint32_t wy, std::uint32_t sx, std::uint32_t si, std::uint32_t sy,
    float *py) {
  const std::uint32_t t = IDX;
  const std::uint32_t ox = blockIdx.y * sx + pi[blockIdx.y * si] * wy;
  const std::uint32_t oy = blockIdx.y * sy;
  if (t < sy) py[oy + t] = px[ox + (t / wy) * wx + (t % wy)];
}

__global__ void slice_fw_dev(
    const float *px, std::uint32_t span, std::uint32_t skip, std::uint32_t size, float *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = px[(i / span) * skip + (i % span)];
}

__global__ void concat_fw_dev(
    const float *px, std::uint32_t span, std::uint32_t skip, std::uint32_t x_size,
    std::uint32_t y_size, float *py) {
  const std::uint32_t i = IDX;
  if (i < y_size) py[(i / span) * skip + (i % span)] = px[i % x_size];
}

__global__ void pick_bw_dev(
    const float *pgy, const std::uint32_t *pi,
    std::uint32_t wx, std::uint32_t wy, std::uint32_t sx, std::uint32_t si, std::uint32_t sy,
    float *pgx) {
  const std::uint32_t t = IDX;
  const std::uint32_t ox = blockIdx.y * sx + pi[blockIdx.y * si] * wy;
  const std::uint32_t oy = blockIdx.y * sy;
  if (t < sy) ::atomicAdd(pgx + ox + (t / wy) * wx + (t % wy), pgy[oy + t]);
}

__global__ void slice_bw_dev(
    const float *pgy, std::uint32_t wx, std::uint32_t wy, std::uint32_t nx, std::uint32_t ny,
    float *pgx) {
  const std::uint32_t i = IDX;
  if (i < wy * ::max(nx, ny)) {
    ::atomicAdd(
        pgx + ((i / wy) * wx + (i % wy)) % (wx * nx), pgy[i % (wy * ny)]);
  }
}

#define CUDADEV_KERNEL_FW_X(name, op) \
__global__ void name##_fw_dev(const float *px, std::uint32_t size, float *py) { \
  const std::uint32_t i = IDX; \
  if (i < size) py[i] = (op); \
}

#define CUDADEV_KERNEL_BW_X(name, op) \
__global__ void name##_bw_dev( \
    const float *px, const float *py, const float *pgy, std::uint32_t size, \
    float *pgx) { \
  static_cast<void>(px); \
  static_cast<void>(py); \
  const std::uint32_t i = IDX; \
  if (i < size) pgx[i] += (op); \
}

#define CUDADEV_KERNEL_FW_X_CONST(name, op) \
__global__ void name##_fw_dev( \
    const float *px, float k, std::uint32_t size, float *py) { \
  const std::uint32_t i = IDX; \
  if (i < size) py[i] = (op); \
}

#define CUDADEV_KERNEL_BW_X_CONST(name, op) \
__global__ void name##_bw_dev( \
    const float *px, const float *py, const float *pgy, float k, \
    std::uint32_t size, float *pgx) { \
  static_cast<void>(px); \
  static_cast<void>(py); \
  const std::uint32_t i = IDX; \
  if (i < size) pgx[i] += (op); \
}

#define CUDADEV_KERNEL_FW_X_SCALAR_R(name, op) \
__global__ void name##_fw_dev( \
    const float *px, const float *pk, std::uint32_t size, std::uint32_t mbx, \
    std::uint32_t mbk, float *py) { \
  const std::uint32_t i = IDX; \
  const std::uint32_t shift = blockIdx.y * size; \
  if (i < size) py[i + shift] = op(px[i + mbx * shift], pk[mbk * blockIdx.y]); \
}

#define CUDADEV_KERNEL_FW_X_SCALAR_L(name, op) \
__global__ void name##_fw_dev( \
    const float *px, const float *pk, std::uint32_t size, std::uint32_t mbx, \
    std::uint32_t mbk, float *py) { \
  const std::uint32_t i = IDX; \
  const std::uint32_t shift = blockIdx.y * size; \
  if (i < size) py[i + shift] = op(pk[mbk * blockIdx.y], px[i + mbx * shift]); \
}

#define CUDADEV_KERNEL_FW_AB(name, op) \
__global__ void name##_fw_dev( \
    const float *pa, const float *pb, std::uint32_t size, std::uint32_t mba, \
    std::uint32_t mbb, float *py) { \
  const std::uint32_t i = IDX; \
  const std::uint32_t shift = blockIdx.y * size; \
  if (i < size) py[i + shift] = op(pa[i + mba * shift], pb[i + mbb * shift]); \
}

CUDADEV_KERNEL_FW_X(negate, -px[i]);
CUDADEV_KERNEL_FW_X(sqrt, ::__fsqrt_rn(px[i]));
CUDADEV_KERNEL_FW_X(exp, ::expf(px[i]));
CUDADEV_KERNEL_FW_X(log, ::logf(px[i]));
CUDADEV_KERNEL_FW_X(tanh, ::tanhf(px[i]));
CUDADEV_KERNEL_FW_X(sigmoid, .5f + .5f * ::tanhf(.5f * px[i]));
CUDADEV_KERNEL_FW_X(
    softplus, ::fmaxf(px[i], .0f) + ::logf(1.f + ::expf(-::fabs(px[i]))));
CUDADEV_KERNEL_FW_X(sin, ::sinf(px[i]));
CUDADEV_KERNEL_FW_X(cos, ::cosf(px[i]));
CUDADEV_KERNEL_FW_X(tan, ::tanf(px[i]));

CUDADEV_KERNEL_BW_X(sqrt, .5f * pgy[i] / py[i]);
CUDADEV_KERNEL_BW_X(exp, py[i] * pgy[i]);
CUDADEV_KERNEL_BW_X(log, pgy[i] / px[i]);
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
CUDADEV_KERNEL_FW_X_CONST(pow_const_r, ::powf(px[i], k));
CUDADEV_KERNEL_FW_X_CONST(pow_const_l, ::powf(k, px[i]));
CUDADEV_KERNEL_FW_X_CONST(prelu, ::fmaxf(px[i], .0f) + k * ::fminf(px[i], .0f));
CUDADEV_KERNEL_FW_X_CONST(
    elu, ::fmaxf(px[i], .0f) + k * (::expf(::fminf(px[i], .0f)) - 1.0f));

CUDADEV_KERNEL_BW_X_CONST(add_const, pgy[i]);
CUDADEV_KERNEL_BW_X_CONST(subtract_const_r, pgy[i]);
CUDADEV_KERNEL_BW_X_CONST(subtract_const_l, -pgy[i]);
CUDADEV_KERNEL_BW_X_CONST(multiply_const, k * pgy[i]);
CUDADEV_KERNEL_BW_X_CONST(divide_const_r, pgy[i] / k);
CUDADEV_KERNEL_BW_X_CONST(divide_const_l, -py[i] * pgy[i] / px[i]);
CUDADEV_KERNEL_BW_X_CONST(pow_const_r, k * pgy[i] * py[i] / px[i]);
CUDADEV_KERNEL_BW_X_CONST(pow_const_l, ::logf(k) * pgy[i] * py[i]);
CUDADEV_KERNEL_BW_X_CONST(prelu, pgy[i] * ((px[i] > .0f) + k * (px[i] <= .0f)));
CUDADEV_KERNEL_BW_X_CONST(
    elu, pgy[i] * ((px[i] > .0f) + (py[i] + k) * (px[i] <= .0f)));

CUDADEV_KERNEL_FW_X_SCALAR_R(add_scalar, ::__fadd_rn);
CUDADEV_KERNEL_FW_X_SCALAR_R(subtract_scalar_r, ::__fsub_rn);
CUDADEV_KERNEL_FW_X_SCALAR_L(subtract_scalar_l, ::__fsub_rn);
CUDADEV_KERNEL_FW_X_SCALAR_R(multiply_scalar, ::__fmul_rn);
CUDADEV_KERNEL_FW_X_SCALAR_R(divide_scalar_r, ::__fdiv_rn);
CUDADEV_KERNEL_FW_X_SCALAR_L(divide_scalar_l, ::__fdiv_rn);
CUDADEV_KERNEL_FW_X_SCALAR_R(pow_scalar_r, ::powf);
CUDADEV_KERNEL_FW_X_SCALAR_L(pow_scalar_l, ::powf);

CUDADEV_KERNEL_FW_AB(add, ::__fadd_rn);
CUDADEV_KERNEL_FW_AB(subtract, ::__fsub_rn);
CUDADEV_KERNEL_FW_AB(multiply, ::__fmul_rn);
CUDADEV_KERNEL_FW_AB(divide, ::__fdiv_rn);
CUDADEV_KERNEL_FW_AB(pow, ::powf);

#undef CUDADEV_KERNEL_FW_X
#undef CUDADEV_KERNEL_BW_X
#undef CUDADEV_KERNEL_FW_X_CONST
#undef CUDADEV_KERNEL_BW_X_CONST
#undef CUDADEV_KERNEL_FW_X_SCALAR_R
#undef CUDADEV_KERNEL_FW_X_SCALAR_L
#undef CUDADEV_KERNEL_FW_AB

__global__ void add_bw_dev(
    const float *, const float *, const float *, const float *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb, float *pga, float *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    ::atomicAdd(pga + i + mba * shift, gy);
    ::atomicAdd(pgb + i + mbb * shift, gy);
  }
}

__global__ void subtract_bw_dev(
    const float *, const float *, const float *, const float *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb, float *pga, float *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    ::atomicAdd(pga + i + mba * shift, gy);
    ::atomicAdd(pgb + i + mbb * shift, -gy);
  }
}

__global__ void multiply_bw_dev(
    const float *pa, const float *pb, const float *, const float *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb, float *pga, float *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const float gy = pgy[i + shift];
    const std::uint32_t a_ofs = i + mba * shift;
    const std::uint32_t b_ofs = i + mbb * shift;
    ::atomicAdd(pga + a_ofs, gy * pb[b_ofs]);
    ::atomicAdd(pgb + b_ofs, gy * pa[a_ofs]);
  }
}

__global__ void divide_bw_dev(
    const float *, const float *pb, const float *py, const float *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb, float *pga, float *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const std::uint32_t b_ofs = i + mbb * shift;
    const std::uint32_t y_ofs = i + shift;
    const float k = pgy[y_ofs] / pb[b_ofs];
    ::atomicAdd(pga + i + mba * shift, k);
    ::atomicAdd(pgb + b_ofs, -k * py[y_ofs]);
  }
}

__global__ void pow_bw_dev(
    const float *pa, const float *pb, const float *py, const float *pgy,
    std::uint32_t size, std::uint32_t mba, std::uint32_t mbb, float *pga, float *pgb) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) {
    const std::uint32_t a_ofs = i + mba * shift;
    const std::uint32_t b_ofs = i + mbb * shift;
    const std::uint32_t y_ofs = i + shift;
    const float k = pgy[y_ofs] * py[y_ofs];
    ::atomicAdd(pga + a_ofs, k * pb[b_ofs] / pa[a_ofs]);
    ::atomicAdd(pgb + b_ofs, k * ::logf(pa[a_ofs]));
  }
}

__global__ void transpose_fw_dev(
    const float *px, std::uint32_t rows, std::uint32_t cols, float *py) {
  const std::uint32_t i = IDX;
  const std::uint32_t j = IDY;
  std::uint32_t ofs = blockIdx.z * rows * cols;
  if (i < rows && j < cols) py[ofs + j + i * cols] = px[ofs + i + j * rows];
}

__global__ void transpose_bw_dev(
    const float *py, std::uint32_t rows, std::uint32_t cols, float *px) {
  const std::uint32_t i = IDX;
  const std::uint32_t j = IDY;
  std::uint32_t ofs = blockIdx.z * rows * cols;
  if (i < rows && j < cols) px[ofs + i + j * rows] += py[ofs + j + i * cols];
}

template<std::uint32_t BLOCK_SIZE>
__global__ void sum_fw_dev(
    const float *px, std::uint32_t skip, std::uint32_t n, float *py) {
  __shared__ float temp[BLOCK_SIZE];
  const std::uint32_t bid = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  px += bid % skip + (bid / skip) * skip * n;
  temp[tid] = 0;
  for (std::uint32_t i = tid; i < n; i += BLOCK_SIZE) temp[tid] += px[i * skip];
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

template<std::uint32_t BLOCK_SIZE>
__global__ void logsumexp_fw_dev(
    const float *px, std::uint32_t skip, std::uint32_t n, float *py) {
  __shared__ float temp[BLOCK_SIZE];
  const std::uint32_t bid = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  px += bid % skip + (bid / skip) * skip * n;
  temp[tid] = -1e38;  // NOTE(odashi): Near the minimum of the float.
  for (std::uint32_t i = tid; i < n; i += BLOCK_SIZE) {
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

template<std::uint32_t BLOCK_SIZE>
__global__ void argmax_dev(
    const float *px, std::uint32_t skip, std::uint32_t n, std::uint32_t *py) {
  __shared__ float max_val[BLOCK_SIZE];
  __shared__ std::uint32_t argmax_val[BLOCK_SIZE];
  const std::uint32_t bid = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  px += bid % skip + (bid / skip) * skip * n;
  max_val[tid] = -1e38;  // NOTE(odashi): Near the minimum of the float.
  for (std::uint32_t i = tid; i < n; i += BLOCK_SIZE) {
    const float val = px[i * skip];
    if (val > max_val[tid]) {
      max_val[tid] = val;
      argmax_val[tid] = i;
    }
  }
  __syncthreads();
#define REDUCE(k) \
  if (BLOCK_SIZE >= k << 1) { \
    if (tid < k) { \
      if (max_val[tid + k] > max_val[tid]) { \
        max_val[tid] = max_val[tid + k]; \
        argmax_val[tid] = argmax_val[tid + k]; \
      } \
    } \
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
  if (tid == 0) py[bid] = argmax_val[0];
}

template<std::uint32_t BLOCK_SIZE>
__global__ void argmin_dev(
    const float *px, std::uint32_t skip, std::uint32_t n, std::uint32_t *py) {
  __shared__ float min_val[BLOCK_SIZE];
  __shared__ std::uint32_t argmin_val[BLOCK_SIZE];
  const std::uint32_t bid = blockIdx.x;
  const std::uint32_t tid = threadIdx.x;
  px += bid % skip + (bid / skip) * skip * n;
  min_val[tid] = 1e38;  // NOTE(odashi): Near the maximum of the float.
  for (std::uint32_t i = tid; i < n; i += BLOCK_SIZE) {
    const float val = px[i * skip];
    if (val < min_val[tid]) {
      min_val[tid] = val;
      argmin_val[tid] = i;
    }
  }
  __syncthreads();
#define REDUCE(k) \
  if (BLOCK_SIZE >= k << 1) { \
    if (tid < k) { \
      if (min_val[tid + k] < min_val[tid]) { \
        min_val[tid] = min_val[tid + k]; \
        argmin_val[tid] = argmin_val[tid + k]; \
      } \
    } \
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
  if (tid == 0) py[bid] = argmin_val[0];
}

__global__ void broadcast_fw_dev(
    const float *px, std::uint32_t skip1, std::uint32_t skip2, std::uint32_t size, float *py) {
  const std::uint32_t i = IDX;
  if (i < size) py[i] = px[i % skip1 + (i / skip2) * skip1];
}

__global__ void batch_sum_fw_dev(
    const float *px, std::uint32_t size, std::uint32_t batch, float *py) {
  const std::uint32_t i = IDX;
  if (i < size) {
    float temp = .0f;
    px += i;
    for (std::uint32_t j = 0; j < batch; ++j, px += size) {
      temp += *px;
    }
    py[i] = temp;
  }
}

__global__ void inplace_multiply_const_dev(
    float k, std::uint32_t size, float *px) {
  const std::uint32_t i = IDX;
  if (i < size) px[i] *= k;
}

__global__ void inplace_add_dev(
    const float *px, std::uint32_t size, std::uint32_t mbx, std::uint32_t mby, float *py) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) ::atomicAdd(py + i + mby * shift, px[i + mbx * shift]);
}

__global__ void inplace_subtract_dev(
    const float *px, std::uint32_t size, std::uint32_t mbx, std::uint32_t mby, float *py) {
  const std::uint32_t i = IDX;
  const std::uint32_t shift = blockIdx.y * size;
  if (i < size) ::atomicAdd(py + i + mby * shift, -px[i + mbx * shift]);
}

#undef IDX
#undef IDY

/*
 * CUBLAS initializer/finalizer.
 */
class CUBLASHandle {
private:
  CUBLASHandle(const CUBLASHandle &) = delete;
  CUBLASHandle(CUBLASHandle &&) = delete;
  CUBLASHandle &operator=(const CUBLASHandle &) = delete;
  CUBLASHandle &operator=(CUBLASHandle &&) = delete;

public:
  explicit CUBLASHandle(std::uint32_t dev_id) {
    CUDA_CALL(::cudaSetDevice(dev_id));
    CUBLAS_CALL(::cublasCreate(&handle_));
    //cerr << "CUBLAS initialized at device " << dev_id << '.' << endl;
  }

  ~CUBLASHandle() {
    CUBLAS_CALL(::cublasDestroy(handle_));
    //cerr << "CUBLAS finalized." << endl;
  }

  ::cublasHandle_t get() const { return handle_; }

private:
  ::cublasHandle_t handle_;
};

/*
 * CURAND initializer/finalizer.
 */
class CURANDHandle {
private:
  CURANDHandle(const CURANDHandle &) = delete;
  CURANDHandle(CURANDHandle &&) = delete;
  CURANDHandle &operator=(const CURANDHandle &) = delete;
  CURANDHandle &operator=(CURANDHandle &&) = delete;

public:
  CURANDHandle(std::uint32_t dev_id, std::uint32_t rng_seed) {
    CUDA_CALL(::cudaSetDevice(dev_id));
    CURAND_CALL(::curandCreateGenerator(&handle_, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(::curandSetPseudoRandomGeneratorSeed(handle_, rng_seed));
    //cerr << "CURAND initialized at device " << dev_id << '.' << endl;
  }

  ~CURANDHandle() {
    CURAND_CALL(::curandDestroyGenerator(handle_));
    //cerr << "CURAND finalized." << endl;
  }

  ::curandGenerator_t get() const { return handle_; }

private:
  ::curandGenerator_t handle_;
};

}  // namespace

namespace primitiv {
namespace devices {

/*
 * Hidden objects of CUDA devices.
 */
struct CUDAInternalState {
  CUDAInternalState(std::uint32_t dev_id, std::uint32_t rng_seed)
    : cublas(dev_id)
    , curand(dev_id, rng_seed)
    , pool(
        [dev_id](std::size_t size) -> void * {  // allocator
          void *ptr;
          CUDA_CALL(::cudaSetDevice(dev_id));
          CUDA_CALL(::cudaMalloc(&ptr, size));
          return ptr;
        },
        [](void *ptr) -> void {  // deleter
          CUDA_CALL(::cudaFree(ptr));
        }) {}
  ::CUBLASHandle cublas;
  ::CURANDHandle curand;
  MemoryPool pool;
  ::cudaDeviceProp prop;
};

std::uint32_t CUDA::num_devices() {
  int ret;
  CUDA_CALL(::cudaGetDeviceCount(&ret));
  return ret;
}

void CUDA::assert_support(std::uint32_t device_id) {
  if (device_id >= num_devices()) {
    THROW_ERROR("Invalid device ID: " << device_id);
  }

  ::cudaDeviceProp prop;
  CUDA_CALL(::cudaGetDeviceProperties(&prop, device_id));

  // Checks compute capability
  static const int MIN_CC_MAJOR = 3;
  static const int MIN_CC_MINOR = 0;
  if (prop.major < MIN_CC_MAJOR ||
      (prop.major == MIN_CC_MAJOR && prop.minor < MIN_CC_MINOR)) {
    THROW_ERROR(
        "CUDA Device " << device_id << " does not satisfy the "
        "minimum requirement of the compute capability: "
        << prop.major << '.' << prop.minor << " < "
        << MIN_CC_MAJOR << '.' << MIN_CC_MINOR);
  }

  // Checks other minimum requirements.
#define CHECK_REQUIREMENT(name, value) \
  { \
    if (prop.name < (value)) { \
      THROW_ERROR( \
          "CUDA Device " << device_id \
          << " does not satisfy the minimum requirement by primitiv. " \
          << "property: " << #name << ", " \
          << "value: " << prop.name << ", " \
          << "required at least: " << (value)); \
    } \
  }
#define CHECK_REQUIREMENT_VECTOR(name, index, value) \
  { \
    if (prop.name[index] < (value)) { \
      THROW_ERROR( \
          "CUDA Device " << device_id \
          << " does not satisfy the minimum requirement by primitiv. " \
          << "property: " << #name << "[" << #index << "], " \
          << "value: " << prop.name[index] << ", " \
          << "required at least: " << (value)); \
    } \
  }

  CHECK_REQUIREMENT(totalGlobalMem, 1ull * (1ull << 30));
  CHECK_REQUIREMENT(sharedMemPerBlock, 16ull * (1ull << 10));
  CHECK_REQUIREMENT(maxThreadsPerBlock, 256);
  CHECK_REQUIREMENT_VECTOR(maxThreadsDim, 0, 256);
  CHECK_REQUIREMENT_VECTOR(maxThreadsDim, 1, 16);
  CHECK_REQUIREMENT_VECTOR(maxThreadsDim, 2, 1);
  CHECK_REQUIREMENT_VECTOR(maxGridSize, 0, 32767);
  CHECK_REQUIREMENT_VECTOR(maxGridSize, 1, 32767);
  CHECK_REQUIREMENT_VECTOR(maxGridSize, 2, 32767);

#undef CHECK_REQUIREMENT
#undef CHECK_REQUIREMENT_VECTOR
}

void CUDA::initialize() {
  assert_support(dev_id_);

  // Retrieves device properties.
  ::cudaDeviceProp prop;
  CUDA_CALL(::cudaGetDeviceProperties(&prop, dev_id_));

  // Calculates size of dims to be used in CUDA kernels.
  dim1_x_ = 1;
  while (dim1_x_ < 1024 &&
      dim1_x_ < static_cast<std::uint32_t>(prop.maxThreadsPerBlock)) {
    dim1_x_ <<= 1;
  }
  dim2_y_ = dim1_x_;
  dim2_x_ = 1;
  while (dim2_x_ < dim2_y_) {
    dim2_x_ <<= 1;
    dim2_y_ >>= 1;
  }
  max_batch_ = prop.maxGridSize[1];

  // Initializes additional libraries
  state_.reset(new CUDAInternalState(dev_id_, rng_seed_));
  state_->prop = prop;

  // Initializes the device pointer for integer IDs.
  ids_ptr_ = state_->pool.allocate(sizeof(std::uint32_t) * max_batch_);
}

CUDA::CUDA(std::uint32_t device_id, std::uint32_t rng_seed)
: dev_id_(device_id)
, rng_seed_(rng_seed) {
  initialize();
}

CUDA::CUDA(std::uint32_t device_id) : CUDA(device_id, std::random_device()()) {}

CUDA::~CUDA() {
  // Nothing to do for now.
}

void CUDA::dump_description() const {
  cerr << "Device " << this << endl;
  cerr << "  Type: CUDA" << endl;

  const ::cudaDeviceProp &prop = state_->prop;
  cerr << "  Device ID: " << dev_id_ << endl;
  cerr << "    Name .................. " << prop.name << endl;
  cerr << "    Global memory ......... " << prop.totalGlobalMem << endl;
  cerr << "    Shared memory/block ... " << prop.sharedMemPerBlock << endl;
  cerr << "    Threads/block ......... " << prop.maxThreadsPerBlock << endl;
  cerr << "    Block size ............ " << prop.maxThreadsDim[0] << ", "
                                         << prop.maxThreadsDim[1] << ", "
                                         << prop.maxThreadsDim[2] << endl;
  cerr << "    Grid size ............. " << prop.maxGridSize[0] << ", "
                                         << prop.maxGridSize[1] << ", "
                                         << prop.maxGridSize[2] << endl;
  cerr << "    Compute capability .... " << prop.major << '.'
                                         << prop.minor << endl;
  /*
  cerr << "  Configurations:" << endl;
  cerr << "    1 dim ........... " << dim1_x_ << " threads" << endl;
  cerr << "    2 dims .......... " << dim2_x_ << "x"
                                 << dim2_y_ << " threads" << endl;
  cerr << "    Maximum batch ... " << max_batch_ <<endl;
  */
}

std::shared_ptr<void> CUDA::new_handle(const Shape &shape) {
  return state_->pool.allocate(sizeof(float) * shape.size());
}

#define GRID_SIZE(x, threads) (((x) + (threads) - 1) / (threads))
#define CDATA(x) static_cast<const float *>(get_handle(x))
#define MDATA(x) static_cast<float *>(get_mutable_handle(x))

std::vector<float> CUDA::tensor_to_vector_impl(const Tensor &x) {
  const std::uint32_t size = x.shape().size();
  std::vector<float> ret(size);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        &ret[0], CDATA(x), sizeof(float) * size, cudaMemcpyDeviceToHost));
  return ret;
}

std::vector<std::uint32_t> CUDA::argmax_impl(const Tensor &x, std::uint32_t dim) {
  const Shape &shape = x.shape();
  const std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  const std::uint32_t s = shape.lower_volume(dim);
  std::uint32_t block_size = dim1_x_;
  while (block_size >> 1 >= n) block_size >>= 1;
  std::shared_ptr<void> py = state_->pool.allocate(sizeof(std::uint32_t) * r);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  switch (block_size) {
#define CASE(k) \
    case k: ::argmax_dev<k><<<r, k>>>( \
        CDATA(x), s, n, static_cast<std::uint32_t *>(py.get())); break
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
  std::vector<std::uint32_t> ret(r);
  CUDA_CALL(::cudaMemcpy(
        &ret[0], py.get(), sizeof(std::uint32_t) * r, cudaMemcpyDeviceToHost));
  return ret;
}

std::vector<std::uint32_t> CUDA::argmin_impl(const Tensor &x, std::uint32_t dim) {
  const Shape &shape = x.shape();
  const std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  const std::uint32_t s = shape.lower_volume(dim);
  std::uint32_t block_size = dim1_x_;
  while (block_size >> 1 >= n) block_size >>= 1;
  std::shared_ptr<void> py = state_->pool.allocate(sizeof(std::uint32_t) * r);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  switch (block_size) {
#define CASE(k) \
    case k: ::argmin_dev<k><<<r, k>>>( \
        CDATA(x), s, n, static_cast<std::uint32_t *>(py.get())); break
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
  std::vector<std::uint32_t> ret(r);
  CUDA_CALL(::cudaMemcpy(
        &ret[0], py.get(), sizeof(std::uint32_t) * r, cudaMemcpyDeviceToHost));
  return ret;
}

void CUDA::reset_tensor_impl(float k, Tensor &x) {
  const std::uint32_t size = x.shape().size();
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::set_const_dev<<<num_blocks, dim1_x_>>>(k, size, MDATA(x));
}

void CUDA::reset_tensor_by_array_impl(const float values[], Tensor &x) {
  const std::uint32_t size = x.shape().size();
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        MDATA(x), values, sizeof(float) * size, cudaMemcpyHostToDevice));
}

void CUDA::copy_tensor_impl(const Tensor &x, Tensor &y) {
  switch (x.device().type()) {
    case Device::DeviceType::NAIVE:
      reset_tensor_by_array(CDATA(x), y);
      break;
    case Device::DeviceType::CUDA:
      CUDA_CALL(::cudaSetDevice(dev_id_));
      // NOTE(odashi):
      // If source/destination devices use the unified memory space on the 64
      // bits machine, we can perform ::cudaMemcpy to copy data beyond devices.
      CUDA_CALL(::cudaMemcpyAsync(
            MDATA(y), CDATA(x),
            sizeof(float) * x.shape().size(),
            cudaMemcpyDeviceToDevice, 0));
      break;
    default:
      reset_tensor_by_vector(x.to_vector(), y);
  }
}

void CUDA::identity_impl(Tensor &y) {
  const std::uint32_t size = y.shape().size();
  const std::uint32_t skip = y.shape()[0] + 1;
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::set_identity_dev<<<num_blocks, dim1_x_>>>(size, skip, MDATA(y));
}

void CUDA::random_bernoulli_impl(float p, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateUniform(state_->curand.get(), MDATA(y), size));
  ::rand_bernoulli_dev<<<num_blocks, dim1_x_>>>(p, size, MDATA(y));
}

void CUDA::random_uniform_impl(float lower, float upper, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_);
  const float scale = upper - lower;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateUniform(state_->curand.get(), MDATA(y), size));
  ::rand_affine_dev<<<num_blocks, dim1_x_>>>(lower, scale, size, MDATA(y));
}

void CUDA::random_normal_impl(float mean, float sd, Tensor &y) {
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateNormal(
        state_->curand.get(), MDATA(y), y.shape().size(), mean, sd));
}

void CUDA::random_log_normal_impl(float mean, float sd, Tensor &y) {
  CUDA_CALL(::cudaSetDevice(dev_id_));
  CURAND_CALL(::curandGenerateLogNormal(
        state_->curand.get(), MDATA(y), y.shape().size(), mean, sd));
}

void CUDA::pick_fw_impl(
    const Tensor &x, const std::vector<std::uint32_t> &ids, std::uint32_t dim,
    Tensor &y) {
  const std::uint32_t wy = y.shape().lower_volume(dim);
  const std::uint32_t sy = y.shape().volume();
  const std::uint32_t g1 = GRID_SIZE(sy, dim1_x_);
  const std::uint32_t bs = y.shape().batch();

  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        ids_ptr_.get(), ids.data(), sizeof(std::uint32_t) * ids.size(),
        cudaMemcpyHostToDevice));
  ::pick_fw_dev<<<dim3(g1, bs), dim1_x_>>>(
      CDATA(x), static_cast<const std::uint32_t *>(ids_ptr_.get()),
      wy * x.shape()[dim], wy,
      x.shape().has_batch() * x.shape().volume(), ids.size() > 1, sy,
      MDATA(y));
}

void CUDA::slice_fw_impl(
    const Tensor &x, std::uint32_t dim, std::uint32_t offset, Tensor &y) {
  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t span = base * y.shape()[dim];
  const std::uint32_t skip = base * x.shape()[dim];
  const std::uint32_t size = y.shape().size();
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::slice_fw_dev<<<num_blocks, dim1_x_>>>(
      CDATA(x) + base * offset, span, skip, size, MDATA(y));
}

void CUDA::concat_fw_impl(
    const std::vector<const Tensor *> &xs, std::uint32_t dim, Tensor &y) {
  const std::uint32_t new_bs = y.shape().batch();
  const std::uint32_t base = y.shape().lower_volume(dim);
  const std::uint32_t skip = base * y.shape()[dim];
  std::uint32_t repeat = y.shape().volume() / skip;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  std::uint32_t offset = 0;
  for (const Tensor *x : xs) {
    const std::uint32_t span = base * x->shape()[dim];
    const std::uint32_t x_size = span * repeat * x->shape().batch();
    const std::uint32_t y_size = span * repeat * new_bs;
    const std::uint32_t num_blocks = GRID_SIZE(y_size, dim1_x_);
    ::concat_fw_dev<<<num_blocks, dim1_x_>>>(
       CDATA(*x), span, skip, x_size, y_size, MDATA(y) + offset);
    offset += span;
  }
}

void CUDA::pick_bw_impl(
    const Tensor &gy, const std::vector<std::uint32_t>& ids, std::uint32_t dim,
    Tensor &gx) {
  const std::uint32_t wy = gy.shape().lower_volume(dim);
  const std::uint32_t sy = gy.shape().volume();
  const std::uint32_t g1 = GRID_SIZE(sy, dim1_x_);
  const std::uint32_t bs = gy.shape().batch();

  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        ids_ptr_.get(), ids.data(), sizeof(std::uint32_t) * ids.size(),
        cudaMemcpyHostToDevice));
  ::pick_bw_dev<<<dim3(g1, bs), dim1_x_>>>(
      CDATA(gy), static_cast<const std::uint32_t *>(ids_ptr_.get()),
      wy *gx.shape()[dim], wy,
      gx.shape().has_batch() * gx.shape().volume(), ids.size() > 1, sy,
      MDATA(gx));
}

void CUDA::slice_bw_impl(
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
  const std::uint32_t g1 = GRID_SIZE(wy * std::max(nx, ny), dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::slice_bw_dev<<<g1, dim1_x_>>>(CDATA(gy), wx, wy, nx, ny, MDATA(gx) + ox);
}

#define CUDADEV_FW_X(name) \
void CUDA::name##_fw_impl(const Tensor &x, Tensor &y) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<num_blocks, dim1_x_>>>(CDATA(x), size, MDATA(y)); \
}

#define CUDADEV_BW_X(name) \
void CUDA::name##_bw_impl( \
    const Tensor &x, const Tensor &y, const Tensor &gy, Tensor &gx) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_bw_dev<<<num_blocks, dim1_x_>>>( \
      CDATA(x), CDATA(y), CDATA(gy), size, MDATA(gx)); \
}

#define CUDADEV_FW_X_CONST(name) \
void CUDA::name##_fw_impl(const Tensor &x, float k, Tensor &y) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = GRID_SIZE(size,dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<num_blocks, dim1_x_>>>(CDATA(x), k, size, MDATA(y)); \
}

#define CUDADEV_BW_X_CONST(name) \
void CUDA::name##_bw_impl( \
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) { \
  const std::uint32_t size = x.shape().size(); \
  const std::uint32_t num_blocks = GRID_SIZE(size, dim1_x_); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_bw_dev<<<num_blocks, dim1_x_>>>( \
      CDATA(x), CDATA(y), CDATA(gy), k, size, MDATA(gx)); \
}

#define CUDADEV_FW_X_SCALAR(name) \
void CUDA::name##_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_); \
  const std::uint32_t g2 = y.shape().batch(); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<dim3(g1, g2, 1), dim1_x_>>>( \
      CDATA(x), CDATA(k), size, \
      x.shape().has_batch(), k.shape().has_batch(), MDATA(y)); \
}

#define CUDADEV_FW_AB(name) \
void CUDA::name##_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_); \
  const std::uint32_t g2 = y.shape().batch(); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_fw_dev<<<dim3(g1, g2, 1), dim1_x_>>>( \
      CDATA(a), CDATA(b), size, \
      a.shape().has_batch(), b.shape().has_batch(), MDATA(y)); \
}

#define CUDADEV_BW_AB(name) \
void CUDA::name##_bw_impl( \
    const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy, \
    Tensor &ga, Tensor &gb) { \
  const std::uint32_t size = y.shape().volume(); \
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_); \
  const std::uint32_t g2 = y.shape().batch(); \
  CUDA_CALL(::cudaSetDevice(dev_id_)); \
  ::name##_bw_dev<<<dim3(g1, g2, 1), dim1_x_>>>( \
      CDATA(a), CDATA(b), CDATA(y), CDATA(gy), size, \
      a.shape().has_batch(), b.shape().has_batch(), MDATA(ga), MDATA(gb)); \
}

CUDADEV_FW_X(negate);
CUDADEV_FW_X(sqrt);
CUDADEV_FW_X(exp);
CUDADEV_FW_X(log);
CUDADEV_FW_X(tanh);
CUDADEV_FW_X(sigmoid);
CUDADEV_FW_X(softplus);
CUDADEV_FW_X(sin);
CUDADEV_FW_X(cos);
CUDADEV_FW_X(tan);

CUDADEV_BW_X(sqrt);
CUDADEV_BW_X(exp);
CUDADEV_BW_X(log);
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
CUDADEV_FW_X_CONST(pow_const_r);
CUDADEV_FW_X_CONST(pow_const_l);
CUDADEV_FW_X_CONST(prelu);
CUDADEV_FW_X_CONST(elu);

CUDADEV_BW_X_CONST(add_const);
CUDADEV_BW_X_CONST(subtract_const_r);
CUDADEV_BW_X_CONST(subtract_const_l);
CUDADEV_BW_X_CONST(multiply_const);
CUDADEV_BW_X_CONST(divide_const_r);
CUDADEV_BW_X_CONST(divide_const_l);
CUDADEV_BW_X_CONST(pow_const_r);
CUDADEV_BW_X_CONST(pow_const_l);
CUDADEV_BW_X_CONST(prelu);
CUDADEV_BW_X_CONST(elu);

CUDADEV_FW_X_SCALAR(add_scalar);
CUDADEV_FW_X_SCALAR(subtract_scalar_r);
CUDADEV_FW_X_SCALAR(subtract_scalar_l);
CUDADEV_FW_X_SCALAR(multiply_scalar);
CUDADEV_FW_X_SCALAR(divide_scalar_r);
CUDADEV_FW_X_SCALAR(divide_scalar_l);
CUDADEV_FW_X_SCALAR(pow_scalar_r);
CUDADEV_FW_X_SCALAR(pow_scalar_l);

CUDADEV_FW_AB(add);
CUDADEV_FW_AB(subtract);
CUDADEV_FW_AB(multiply);
CUDADEV_FW_AB(divide);
CUDADEV_FW_AB(pow);

CUDADEV_BW_AB(add);
CUDADEV_BW_AB(subtract);
CUDADEV_BW_AB(multiply);
CUDADEV_BW_AB(divide);
CUDADEV_BW_AB(pow);

#undef CUDADEV_FW_X
#undef CUDADEV_BW_X
#undef CUDADEV_FW_X_CONST
#undef CUDADEV_BW_X_CONST
#undef CUDADEV_FW_X_SCALAR
#undef CUDADEV_FW_AB
#undef CUDADEV_BW_AB

void CUDA::transpose_fw_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t rows = x.shape()[0];
  const std::uint32_t cols = x.shape()[1];
  const std::uint32_t bs = x.shape().batch();
  const std::uint32_t g1 = GRID_SIZE(rows, dim2_x_);
  const std::uint32_t g2 = GRID_SIZE(cols, dim2_y_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::transpose_fw_dev<<<dim3(g1, g2, bs), dim3(dim2_x_, dim2_y_, 1)>>>(
      CDATA(x), rows, cols, MDATA(y));
}

void CUDA::matmul_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1];
  const std::uint32_t dk = b.shape()[1];
  float alpha = 1.;
  float beta = 0.;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  if (a.shape().has_batch()) {
    // Do gemm multiple times.
    const std::uint32_t a_skip = di * dj;
    const std::uint32_t b_skip = b.shape().has_batch() * dj * dk;
    const std::uint32_t y_skip = di * dk;
    const std::uint32_t bs = a.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      CUBLAS_CALL(::cublasSgemm(
            state_->cublas.get(), ::CUBLAS_OP_N, ::CUBLAS_OP_N,
            di, dk, dj,
            &alpha, CDATA(a) + n * a_skip, di, CDATA(b) + n * b_skip, dj,
            &beta, MDATA(y) + n * y_skip, di));
    }
  } else {
    // Do gemm only once to calculate the product with a combined matrix.
    CUBLAS_CALL(::cublasSgemm(
          state_->cublas.get(), ::CUBLAS_OP_N, ::CUBLAS_OP_N,
          di, dk * b.shape().batch(), dj,
          &alpha, CDATA(a), di, CDATA(b), dj,
          &beta, MDATA(y), di));
  }
}

void CUDA::transpose_bw_impl(
    const Tensor &, const Tensor &, const Tensor &gy, Tensor &gx) {
  const std::uint32_t rows = gx.shape()[0];
  const std::uint32_t cols = gx.shape()[1];
  const std::uint32_t bs = gx.shape().batch();
  const std::uint32_t g1 = GRID_SIZE(rows, dim2_x_);
  const std::uint32_t g2 = GRID_SIZE(cols, dim2_y_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::transpose_bw_dev<<<dim3(g1, g2, bs), dim3(dim2_x_, dim2_y_, 1)>>>(
      CDATA(gy), rows, cols, MDATA(gx));
}

void CUDA::matmul_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  // ga += gy . b^T
  // gb += a^T . gy
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1];
  const std::uint32_t dk = b.shape()[1];
  float alpha = 1.;
  float beta = 1.;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  if (a.shape().has_batch()) {
    // Do gemm multiple times.
    const std::uint32_t a_skip = di * dj;
    const std::uint32_t b_skip = b.shape().has_batch() * dj * dk;
    const std::uint32_t y_skip = di * dk;
    const std::uint32_t bs = a.shape().batch();
    for (std::uint32_t n = 0; n < bs; ++n) {
      CUBLAS_CALL(::cublasSgemm(
            state_->cublas.get(), ::CUBLAS_OP_N, ::CUBLAS_OP_T,
            di, dj, dk,
            &alpha, CDATA(gy) + n * y_skip, di, CDATA(b) + n * b_skip, dj,
            &beta, MDATA(ga) + n * a_skip, di));
      CUBLAS_CALL(::cublasSgemm(
            state_->cublas.get(), ::CUBLAS_OP_T, ::CUBLAS_OP_N,
            dj, dk, di,
            &alpha, CDATA(a) + n * a_skip, di, CDATA(gy) + n * y_skip, di,
            &beta, MDATA(gb) + n * b_skip, dj));
    }
  } else {
    // Do gemm only once to calculate the product with a combined matrix.
    CUBLAS_CALL(::cublasSgemm(
          state_->cublas.get(), ::CUBLAS_OP_N, ::CUBLAS_OP_T,
          di, dj, dk * b.shape().batch(),
          &alpha, CDATA(gy), di, CDATA(b), dj,
          &beta, MDATA(ga), di));
    CUBLAS_CALL(::cublasSgemm(
          state_->cublas.get(), ::CUBLAS_OP_T, ::CUBLAS_OP_N,
          dj, dk * b.shape().batch(), di,
          &alpha, CDATA(a), di, CDATA(gy), di,
          &beta, MDATA(gb), dj));
  }
}

void CUDA::sum_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t block_size = dim1_x_;
  while (block_size >> 1 >= n) block_size >>= 1;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  switch (block_size) {
#define CASE(k) \
    case k: ::sum_fw_dev<k><<<r, k>>>(CDATA(x), s, n, MDATA(y)); break
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

void CUDA::logsumexp_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t block_size = dim1_x_;
  while (block_size >> 1 >= n) block_size >>= 1;
  CUDA_CALL(::cudaSetDevice(dev_id_));
  switch (block_size) {
#define CASE(k) \
    case k: ::logsumexp_fw_dev<k><<<r, k>>>(CDATA(x), s, n, MDATA(y)); break
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

void CUDA::broadcast_fw_impl(
    const Tensor &x, std::uint32_t dim, std::uint32_t size, Tensor &y) {
  const std::uint32_t skip1 = y.shape().lower_volume(dim);
  const std::uint32_t skip2 = skip1 * size;
  const std::uint32_t total = y.shape().size();
  const std::uint32_t g1 = GRID_SIZE(total, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::broadcast_fw_dev<<<g1, dim1_x_>>>(CDATA(x), skip1, skip2, total, MDATA(y));
}

void CUDA::batch_sum_fw_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t size = y.shape().size();
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::batch_sum_fw_dev<<<g1, dim1_x_>>>(
      CDATA(x), size, x.shape().batch(), MDATA(y));
}

void CUDA::inplace_multiply_const_impl(float k, Tensor &x) {
  const std::uint32_t size = x.shape().size();
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_);
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::inplace_multiply_const_dev<<<g1, dim1_x_>>>(k, size, MDATA(x));
}

void CUDA::inplace_add_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t size = y.shape().volume();
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_);
  const std::uint32_t bs = std::max(x.shape().batch(), y.shape().batch());
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::inplace_add_dev<<<dim3(g1, bs, 1), dim1_x_>>>(
      CDATA(x), size, x.shape().has_batch(), y.shape().has_batch(), MDATA(y));
}

void CUDA::inplace_subtract_impl(const Tensor &x, Tensor &y) {
  const std::uint32_t size = y.shape().volume();
  const std::uint32_t g1 = GRID_SIZE(size, dim1_x_);
  const std::uint32_t bs = std::max(x.shape().batch(), y.shape().batch());
  CUDA_CALL(::cudaSetDevice(dev_id_));
  ::inplace_subtract_dev<<<dim3(g1, bs, 1), dim1_x_>>>(
      CDATA(x), size, x.shape().has_batch(), y.shape().has_batch(), MDATA(y));
}

}  // namespace devices
}  // namespace primitiv
