#include <primitiv/config.h>

#include <primitiv/cuda_device.h>
#include <primitiv/internal/cuda_utils.h>
#include <primitiv/device_ops/cuda/common.h>

namespace primitiv {
namespace devices {

void CUDA::batch_concat_fw_impl(
    const std::vector<const Tensor *> &xs, Tensor &y) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

void CUDA::batch_concat_bw_impl(
    const std::vector<const Tensor *> &xs, const Tensor &y, const Tensor &gy,
    const std::vector<Tensor *> &gxs) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

}  // namespace devices
}  // namespace primitiv
