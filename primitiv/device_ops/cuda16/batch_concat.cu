#include <primitiv/config.h>

#include <primitiv/cuda16_device.h>
#include <primitiv/internal/cuda_utils.h>
#include <primitiv/device_ops/cuda16/common.h>

namespace primitiv {
namespace devices {

void CUDA16::batch_concat_fw_impl(
    const std::vector<const Tensor *> &xs, Tensor &y) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

void CUDA16::batch_concat_bw_impl(
    const std::vector<const Tensor *> &xs, const Tensor &y, const Tensor &gy,
    const std::vector<Tensor *> &gxs) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

}  // namespace devices
}  // namespace primitiv
