#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

std::vector<float> OpenCL::tensor_to_vector_impl(const Tensor &x) {
  const std::uint32_t size = x.shape().size();
  std::vector<float> ret(size);
  ::read_buffer(state_->queue, CDATA(x), ret.data(), size);
  return ret;
}

}  // namespace devices
}  // namespace primitiv
