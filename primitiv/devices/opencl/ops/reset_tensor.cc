#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::reset_tensor_impl(float k, Tensor &x) {
  const std::uint32_t size = x.shape().size();
  state_->queue.enqueueFillBuffer<float>(MDATA(x), k, 0, sizeof(float) * size);
}

}  // namespace devices
}  // namespace primitiv
