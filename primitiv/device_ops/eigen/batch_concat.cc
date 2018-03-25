#include <primitiv/config.h>

#include <primitiv/eigen_device.h>
#include <primitiv/device_ops/eigen/common.h>

namespace primitiv {
namespace devices {

void Eigen::batch_concat_fw_impl(
    const std::vector<const Tensor *> &xs, Tensor &y) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

void Eigen::batch_concat_bw_impl(
    const std::vector<const Tensor *> &xs, const Tensor &y, const Tensor &gy,
    const std::vector<Tensor *> &gxs) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

}  // namespace devices
}  // namespace primitiv
