#include <primitiv/config.h>

#include <cstring>

#include <primitiv/naive_device.h>
#include <primitiv/device_ops/common_naive.h>

namespace primitiv {
namespace devices {

std::vector<float> Naive::tensor_to_vector_impl(const Tensor &x) {
  const std::uint32_t num_elements = x.shape().size();
  std::vector<float> ret(num_elements);
  std::memcpy(&ret[0], CDATA(x), sizeof(float) * num_elements);
  return ret;
}

}  // namespace devices
}  // namespace primitiv
