#include <primitiv/config.h>

#include <cstring>

#include <primitiv/cuda16_device.h>
#include <primitiv/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace {

#if CUDART_VERSION >= 9000
__global__ void set_gemm_ptrs(
    const half *pa, const half *pb, const half *py,
    std::uint32_t na, std::uint32_t nb, std::uint32_t ny, std::uint32_t bs,
    const half **ptrs) {
  const std::uint32_t i = IDX;
  if (i < bs) {
    ptrs[i] = pa + i * na;
    ptrs[i + bs] = pb + i * nb;
    ptrs[i + 2 * bs] = py + i * ny;
  }
}
#endif  // CUDART_VERSION

inline half half_zero() {
  static_assert(sizeof(half) == sizeof(std::uint16_t), "");
  constexpr std::uint16_t repr = 0x0000;
  half ret;
  std::memcpy(&ret, &repr, sizeof(half));
  return ret;
}

inline half half_one() {
  static_assert(sizeof(half) == sizeof(std::uint16_t), "");
  constexpr std::uint16_t repr = 0x3c00;
  half ret;
  std::memcpy(&ret, &repr, sizeof(half));
  return ret;
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::matmul_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1];
  const std::uint32_t dk = b.shape()[1];
  half alpha = ::half_one();
  half beta = ::half_zero();

  CUDA_CALL(::cudaSetDevice(dev_id_));

  if (a.shape().has_batch()) {
    // Do gemm multiple times.
    const half *pa = CDATA(half, a);
    const half *pb = CDATA(half, b);
    half *py = MDATA(half, y);
    const std::uint32_t na = di * dj;
    const std::uint32_t nb = b.shape().has_batch() * dj * dk;
    const std::uint32_t ny = di * dk;
    const std::uint32_t bs = a.shape().batch();

#if CUDART_VERSION >= 9000

    std::shared_ptr<void> ptrs = state_->pool.allocate(3 * bs * sizeof(void *));
    const half **fptrs = static_cast<const half **>(ptrs.get());

    const std::uint32_t gs = GRID_SIZE(bs, dim1_x_);

    ::set_gemm_ptrs<<<gs, dim1_x_>>>(pa, pb, py, na, nb, ny, bs, fptrs);
    CUBLAS_CALL(::cublasHgemmBatched(
          state_->cublas.get(), ::CUBLAS_OP_N, ::CUBLAS_OP_N,
          di, dk, dj,
          &alpha, fptrs, di, fptrs + bs, dj,
          &beta, const_cast<half **>(fptrs) + 2 * bs, di,
          bs));

#else  // CUDART_VERSION < 9000

    for (std::uint32_t n = 0; n < bs; ++n) {
      CUBLAS_CALL(::cublasHgemm(
            state_->cublas.get(), ::CUBLAS_OP_N, ::CUBLAS_OP_N,
            di, dk, dj,
            &alpha, pa + n * na, di, pb + n * nb, dj,
            &beta, py + n * ny, di));
    }

#endif  // CUDART_VERSION

  } else {
    // Do gemm only once to calculate the product with a combined matrix.
    CUBLAS_CALL(::cublasHgemm(
          state_->cublas.get(), ::CUBLAS_OP_N, ::CUBLAS_OP_N,
          di, dk * b.shape().batch(), dj,
          &alpha, CDATA(half, a), di, CDATA(half, b), dj,
          &beta, MDATA(half, y), di));
  }
}

void CUDA16::matmul_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  // ga += gy . b^T
  // gb += a^T . gy
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1];
  const std::uint32_t dk = b.shape()[1];
  half alpha = ::half_one();
  half beta = ::half_one();

  CUDA_CALL(::cudaSetDevice(dev_id_));

  if (a.shape().has_batch()) {
    // Do gemm multiple times.
    const half *pa = CDATA(half, a);
    const half *pb = CDATA(half, b);
    const half *pgy = CDATA(half, gy);
    half *pga = MDATA(half, ga);
    half *pgb = MDATA(half, gb);
    const std::uint32_t na = di * dj;
    const std::uint32_t nb = b.shape().has_batch() * dj * dk;
    const std::uint32_t ny = di * dk;
    const std::uint32_t bs = a.shape().batch();

#if CUDART_VERSION >= 9000

    std::shared_ptr<void> ptrs = state_->pool.allocate(3 * bs * sizeof(void *));
    const half **fptrs = static_cast<const half **>(ptrs.get());

    const std::uint32_t gs = GRID_SIZE(bs, dim1_x_);

    ::set_gemm_ptrs<<<gs, dim1_x_>>>(pgy, pb, pga, ny, nb, na, bs, fptrs);
    CUBLAS_CALL(::cublasHgemmBatched(
          state_->cublas.get(), ::CUBLAS_OP_N, ::CUBLAS_OP_T,
          di, dj, dk,
          &alpha, fptrs, di, fptrs + bs, dj,
          &beta, const_cast<half **>(fptrs) + 2 * bs, di,
          bs));

    if (nb > 0 /* `b` has minibatch */) {
      ::set_gemm_ptrs<<<gs, dim1_x_>>>(pa, pgy, pgb, na, ny, nb, bs, fptrs);
      CUBLAS_CALL(::cublasHgemmBatched(
            state_->cublas.get(), ::CUBLAS_OP_T, ::CUBLAS_OP_N,
            dj, dk, di,
            &alpha, fptrs, di, fptrs + bs, di,
            &beta, const_cast<half **>(fptrs) + 2 * bs, dj,
            bs));
    } else {
      // NOTE(odashi):
      // `cublasHgemmBatched` can not be used due to a data race against
      // shared values in `b` by multiple GEMM operations.
      for (std::uint32_t batch = 0; batch < bs; ++batch) {
        CUBLAS_CALL(::cublasHgemm(
              state_->cublas.get(), ::CUBLAS_OP_T, ::CUBLAS_OP_N,
              dj, dk, di,
              &alpha, pa + batch * na, di, pgy + batch * ny, di,
              &beta, pgb, dj));
      }
    }

#else  // CUDART_VERSION < 9000

    for (std::uint32_t n = 0; n < bs; ++n) {
      CUBLAS_CALL(::cublasHgemm(
            state_->cublas.get(), ::CUBLAS_OP_N, ::CUBLAS_OP_T,
            di, dj, dk,
            &alpha, pgy + n * ny, di, pb + n * nb, dj,
            &beta, pga + n * na, di));
      CUBLAS_CALL(::cublasHgemm(
            state_->cublas.get(), ::CUBLAS_OP_T, ::CUBLAS_OP_N,
            dj, dk, di,
            &alpha, pa + n * na, di, pgy + n * ny, di,
            &beta, pgb + n * nb, dj));
    }

#endif  // CURART_VERSION

  } else {
    // Do gemm only once to calculate the product with a combined matrix.
    CUBLAS_CALL(::cublasHgemm(
          state_->cublas.get(), ::CUBLAS_OP_N, ::CUBLAS_OP_T,
          di, dj, dk * b.shape().batch(),
          &alpha, CDATA(half, gy), di, CDATA(half, b), dj,
          &beta, MDATA(half, ga), di));
    CUBLAS_CALL(::cublasHgemm(
          state_->cublas.get(), ::CUBLAS_OP_T, ::CUBLAS_OP_N,
          dj, dk * b.shape().batch(), di,
          &alpha, CDATA(half, a), di, CDATA(half, gy), di,
          &beta, MDATA(half, gb), dj));
  }
}

}  // namespace devices
}  // namespace primitiv
