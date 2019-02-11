#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

#include <clblast.h>

namespace primitiv {
namespace devices {

void OpenCL::matmul_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1];
  const std::uint32_t dk = b.shape()[1];
  if (a.shape().has_batch()) {
    // Do gemm multiple times.
    const std::uint32_t a_skip = di * dj;
    const std::uint32_t b_skip = b.shape().has_batch() * dj * dk;
    const std::uint32_t y_skip = di * dk;
    const std::uint32_t bs = a.shape().batch();
    const std::vector<float> alphas(bs, 1.);
    const std::vector<float> betas(bs, 0.);
    std::vector<std::size_t> a_offsets(bs);
    std::vector<std::size_t> b_offsets(bs);
    std::vector<std::size_t> y_offsets(bs);
    for (std::uint32_t n = 0; n < bs; ++n) {
      a_offsets[n] = n * a_skip;
      b_offsets[n] = n * b_skip;
      y_offsets[n] = n * y_skip;
    }
    clblast::GemmBatched(
      clblast::Layout::kColMajor,
      clblast::Transpose::kNo, clblast::Transpose::kNo,
      di, dk, dj,
      alphas.data(),
      CDATA(a)(), a_offsets.data(), di,
      CDATA(b)(), b_offsets.data(), dj,
      betas.data(),
      MDATA(y)(), y_offsets.data(), di,
      bs,
      &state_->queue(), nullptr);
  } else {
    // Do gemm only once to calculate the product with a combined matrix.
    const float alpha = 1.;
    const float beta = 0.;
    clblast::Gemm(
      clblast::Layout::kColMajor,
      clblast::Transpose::kNo, clblast::Transpose::kNo,
      di, dk * b.shape().batch(), dj,
      alpha,
      CDATA(a)(), 0, di,
      CDATA(b)(), 0, dj,
      beta,
      MDATA(y)(), 0, di,
      &state_->queue(), nullptr);
  }
}

void OpenCL::matmul_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  // ga += gy . b^T
  // gb += a^T . gy
  const std::uint32_t di = a.shape()[0];
  const std::uint32_t dj = a.shape()[1];
  const std::uint32_t dk = b.shape()[1];
  if (a.shape().has_batch()) {
    // Do gemm multiple times.
    const std::uint32_t a_skip = di * dj;
    const std::uint32_t b_skip = b.shape().has_batch() * dj * dk;
    const std::uint32_t y_skip = di * dk;
    const std::uint32_t bs = a.shape().batch();
    const std::vector<float> alphas(bs, 1.);
    const std::vector<float> betas(bs, 1.);
    std::vector<std::size_t> a_offsets(bs);
    std::vector<std::size_t> b_offsets(bs);
    std::vector<std::size_t> y_offsets(bs);
    for (std::uint32_t n = 0; n < bs; ++n) {
      a_offsets[n] = n * a_skip;
      b_offsets[n] = n * b_skip;
      y_offsets[n] = n * y_skip;
    }
    clblast::GemmBatched(
      clblast::Layout::kColMajor,
      clblast::Transpose::kNo, clblast::Transpose::kYes,
      di, dj, dk,
      alphas.data(),
      CDATA(gy)(), y_offsets.data(), di,
      CDATA(b)(), b_offsets.data(), dj,
      betas.data(),
      MDATA(ga)(), a_offsets.data(), di,
      bs,
      &state_->queue(), nullptr);
    if (b_skip > 0) {
      clblast::GemmBatched(
        clblast::Layout::kColMajor,
        clblast::Transpose::kYes, clblast::Transpose::kNo,
        dj, dk, di,
        alphas.data(),
        CDATA(a)(), a_offsets.data(), di,
        CDATA(gy)(), y_offsets.data(), di,
        betas.data(),
        MDATA(gb)(), b_offsets.data(), dj,
        bs,
        &state_->queue(), nullptr);
    } else {
      // NOTE(vbkaisetsu):
      // `clblast::GemmBatched` can not be used due to a data race against
      // shared values in `b` by multiple GEMM operations.
      const float alpha = 1.;
      const float beta = 1.;
      for (std::uint32_t n = 0; n < bs; ++n) {
        clblast::Gemm(
          clblast::Layout::kColMajor,
          clblast::Transpose::kYes, clblast::Transpose::kNo,
          dj, dk, di,
          alpha,
          CDATA(a)(), n * a_skip, di,
          CDATA(gy)(), n * y_skip, di,
          beta,
          MDATA(gb)(), n * b_skip, dj,
          &state_->queue(), nullptr);
      }
    }
  } else {
    // Do gemm only once to calculate the product with a combined matrix.
    const float alpha = 1.;
    const float beta = 1.;
    clblast::Gemm(
      clblast::Layout::kColMajor,
      clblast::Transpose::kNo, clblast::Transpose::kYes,
      di, dj, dk * b.shape().batch(),
      alpha,
      CDATA(gy)(), 0, di,
      CDATA(b)(), 0, dj,
      beta,
      MDATA(ga)(), 0, di,
      &state_->queue(), nullptr);
    clblast::Gemm(
      clblast::Layout::kColMajor,
      clblast::Transpose::kYes, clblast::Transpose::kNo,
      dj, dk * b.shape().batch(), di,
      alpha,
      CDATA(a)(), 0, di,
      CDATA(gy)(), 0, di,
      beta,
      MDATA(gb)(), 0, dj,
      &state_->queue(), nullptr);
  }
}

}  // namespace devices
}  // namespace primitiv
