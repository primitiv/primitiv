#include <primitiv/config.h>

#include <primitiv/devices/cuda/device.h>
#include <primitiv/devices/cuda/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void set_gemm_ptrs(
    const float *pa, const float *pb, const float *py,
    std::uint32_t na, std::uint32_t nb, std::uint32_t ny, std::uint32_t bs,
    const float **ptrs) {
  const std::uint32_t i = IDX;
  if (i < bs) {
    ptrs[i] = pa + i * na;
    ptrs[i + bs] = pb + i * nb;
    ptrs[i + 2 * bs] = py + i * ny;
  }
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA::matmul_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1];
  const std::uint32_t dk = b.shape()[1];
  float alpha = 1.;
  float beta = 0.;

  CUDA_CALL(::cudaSetDevice(dev_id_));

  if (a.shape().has_batch()) {
    // Do gemm multiple times.
    const float *pa = CDATA(a);
    const float *pb = CDATA(b);
    float *py = MDATA(y);
    const std::uint32_t na = di * dj;
    const std::uint32_t nb = b.shape().has_batch() * dj * dk;
    const std::uint32_t ny = di * dk;
    const std::uint32_t bs = a.shape().batch();

    std::shared_ptr<void> ptrs = state_->pool.allocate(3 * bs * sizeof(void *));
    const float **fptrs = static_cast<const float **>(ptrs.get());

    const std::uint32_t gs = GRID_SIZE(bs, dim1_x_);

    ::set_gemm_ptrs<<<gs, dim1_x_>>>(pa, pb, py, na, nb, ny, bs, fptrs);
    CUBLAS_CALL(::cublasSgemmBatched(
          state_->cublas.get(), ::CUBLAS_OP_N, ::CUBLAS_OP_N,
          di, dk, dj,
          &alpha, fptrs, di, fptrs + bs, dj,
          &beta, const_cast<float **>(fptrs) + 2 * bs, di,
          bs));
  } else {
    // Do gemm only once to calculate the product with a combined matrix.
    CUBLAS_CALL(::cublasSgemm(
          state_->cublas.get(), ::CUBLAS_OP_N, ::CUBLAS_OP_N,
          di, dk * b.shape().batch(), dj,
          &alpha, CDATA(a), di, CDATA(b), dj,
          &beta, MDATA(y), di));
  }
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
    const float *pa = CDATA(a);
    const float *pb = CDATA(b);
    const float *pgy = CDATA(gy);
    float *pga = MDATA(ga);
    float *pgb = MDATA(gb);
    const std::uint32_t na = di * dj;
    const std::uint32_t nb = b.shape().has_batch() * dj * dk;
    const std::uint32_t ny = di * dk;
    const std::uint32_t bs = a.shape().batch();

    std::shared_ptr<void> ptrs = state_->pool.allocate(3 * bs * sizeof(void *));
    const float **fptrs = static_cast<const float **>(ptrs.get());

    const std::uint32_t gs = GRID_SIZE(bs, dim1_x_);

    ::set_gemm_ptrs<<<gs, dim1_x_>>>(pgy, pb, pga, ny, nb, na, bs, fptrs);
    CUBLAS_CALL(::cublasSgemmBatched(
          state_->cublas.get(), ::CUBLAS_OP_N, ::CUBLAS_OP_T,
          di, dj, dk,
          &alpha, fptrs, di, fptrs + bs, dj,
          &beta, const_cast<float **>(fptrs) + 2 * bs, di,
          bs));

    if (nb > 0 /* `b` has minibatch */) {
      ::set_gemm_ptrs<<<gs, dim1_x_>>>(pa, pgy, pgb, na, ny, nb, bs, fptrs);
      CUBLAS_CALL(::cublasSgemmBatched(
            state_->cublas.get(), ::CUBLAS_OP_T, ::CUBLAS_OP_N,
            dj, dk, di,
            &alpha, fptrs, di, fptrs + bs, di,
            &beta, const_cast<float **>(fptrs) + 2 * bs, dj,
            bs));
    } else {
      // NOTE(odashi):
      // `cublasSgemmBatched` can not be used due to a data race against
      // shared values in `b` by multiple GEMM operations.
      for (std::uint32_t batch = 0; batch < bs; ++batch) {
        CUBLAS_CALL(::cublasSgemm(
              state_->cublas.get(), ::CUBLAS_OP_T, ::CUBLAS_OP_N,
              dj, dk, di,
              &alpha, pa + batch * na, di, pgy + batch * ny, di,
              &beta, pgb, dj));
      }
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

}  // namespace devices
}  // namespace primitiv
