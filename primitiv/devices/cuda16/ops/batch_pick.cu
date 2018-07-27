#include <primitiv/config.h>

#include <primitiv/devices/cuda16/device.h>
#include <primitiv/devices/cuda16/ops/common.h>
#include <primitiv/internal/cuda/utils.h>

namespace {

__global__ void batch_pick_fw_dev(
    const half *px, const std::uint32_t *pi,
    std::uint32_t si, std::uint32_t sy, half *py) {
  const std::uint32_t t = IDX;
  const std::uint32_t ox = pi[blockIdx.y * si] * sy;
  const std::uint32_t oy = blockIdx.y * sy;
  if (t < sy) py[oy + t] = px[ox + t];
}

DECL_ATOMIC_OP(atomicHAdd, ::__fadd_rn);

__global__ void batch_pick_bw_dev(
    const half *pgy, const std::uint32_t *pi,
    std::uint32_t si, std::uint32_t sy, half *pgx) {
  const std::uint32_t t = IDX;
  const std::uint32_t ox = pi[blockIdx.y * si] * sy;
  const std::uint32_t oy = blockIdx.y * sy;
  if (t < sy) ::atomicHAdd(pgx, ox + t, ::__half2float(pgy[oy + t]));
}

}  // namespace

namespace primitiv {
namespace devices {

void CUDA16::batch_pick_fw_impl(
    const Tensor &x, const std::vector<std::uint32_t> &ids, Tensor &y) {
  const std::uint32_t sy = y.shape().volume();
  const std::uint32_t g1 = GRID_SIZE(sy, dim1_x_);
  const std::uint32_t bs = y.shape().batch();

  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        ids_ptr_.get(), ids.data(), sizeof(std::uint32_t) * ids.size(),
        cudaMemcpyHostToDevice));
  ::batch_pick_fw_dev<<<dim3(g1, bs), dim1_x_>>>(
      CDATA(half, x), static_cast<const std::uint32_t *>(ids_ptr_.get()),
      ids.size() > 1, sy, MDATA(half, y));
}

void CUDA16::batch_pick_bw_impl(
    const Tensor &gy, const std::vector<std::uint32_t>& ids, Tensor &gx) {
  const std::uint32_t sy = gy.shape().volume();
  const std::uint32_t g1 = GRID_SIZE(sy, dim1_x_);
  const std::uint32_t bs = gy.shape().batch();

  CUDA_CALL(::cudaSetDevice(dev_id_));
  CUDA_CALL(::cudaMemcpy(
        ids_ptr_.get(), ids.data(), sizeof(std::uint32_t) * ids.size(),
        cudaMemcpyHostToDevice));
  ::batch_pick_bw_dev<<<dim3(g1, bs), dim1_x_>>>(
      CDATA(half, gy), static_cast<const std::uint32_t *>(ids_ptr_.get()),
      ids.size() > 1, sy, MDATA(half, gx));
}

}  // namespace devices
}  // namespace primitiv
