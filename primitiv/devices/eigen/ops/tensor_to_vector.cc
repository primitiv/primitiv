#include <primitiv/config.h>

#include <cstring>

#include <primitiv/devices/eigen/device.h>
#include <primitiv/devices/eigen/ops/common.h>

namespace primitiv {
namespace devices {

std::vector<float> Eigen::tensor_to_vector_impl(const Tensor &x) {
  const std::uint32_t num_elements = x.shape().size();
  std::vector<float> ret(num_elements);
  std::memcpy(&ret[0], CDATA(x), sizeof(float) * num_elements);
  return ret;
}

}  // namespace devices
}  // namespace primitiv
