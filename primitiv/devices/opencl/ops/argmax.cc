#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

std::vector<std::uint32_t> OpenCL::argmax_impl(
    const Tensor &x, std::uint32_t dim) {
  if (!state_->argmax_kernel.initialized()) {
    state_->initialize_kernel(state_->argmax_kernel, std::string({
#include "primitiv/devices/opencl/kernels/argmax.inc"
    }), "argmax_kernel");
  }
  const Shape &shape = x.shape();
  const std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  const std::uint32_t s = shape.lower_volume(dim);
  std::shared_ptr<void> py = state_->pool.allocate(sizeof(std::uint32_t) * r);
  state_->argmax_kernel.kernel().setArg(0, CDATA(x));
  state_->argmax_kernel.kernel().setArg(1, s);
  state_->argmax_kernel.kernel().setArg(2, n);
  state_->argmax_kernel.kernel().setArg(3, ::get_buffer(py));
  state_->queue.enqueueNDRangeKernel(
      state_->argmax_kernel.kernel(), cl::NullRange,
      cl::NDRange(r * state_->argmax_kernel.group_size()[0]),
      cl::NDRange(state_->argmax_kernel.group_size()[0]));
  std::vector<std::uint32_t> ret(r);
  ::read_buffer(state_->queue, ::get_buffer(py), ret.data(), r);
  return ret;
}

}  // namespace devices
}  // namespace primitiv
