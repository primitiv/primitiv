#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

std::vector<std::uint32_t> OpenCL::argmin_impl(
    const Tensor &x, std::uint32_t dim) {
  const Shape &shape = x.shape();
  const std::uint32_t n = shape[dim];
  const std::uint32_t r = shape.size() / n;
  const std::uint32_t s = shape.lower_volume(dim);
  std::uint32_t group_size = std::min(state_->argmin_group_size, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  std::shared_ptr<void> py = state_->pool.allocate(sizeof(std::uint32_t) * r);
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      state_->argmin_kernel[m].setArg(0, CDATA(x)); \
      state_->argmin_kernel[m].setArg(1, s); \
      state_->argmin_kernel[m].setArg(2, n); \
      state_->argmin_kernel[m].setArg(3, ::get_buffer(py)); \
      state_->queue.enqueueNDRangeKernel( \
          state_->argmin_kernel[m], \
          cl::NullRange, cl::NDRange(r * k), cl::NDRange(k)); \
      break;
    CASE(1024, 10);
    CASE(512, 9);
    CASE(256, 8);
    CASE(128, 7);
    CASE(64, 6);
    CASE(32, 5);
    CASE(16, 4);
    CASE(8, 3);
    CASE(4, 2);
    CASE(2, 1);
    CASE(1, 0);
#undef CASE
  }
  std::vector<std::uint32_t> ret(r);
  ::read_buffer(state_->queue, ::get_buffer(py), ret.data(), r);
  return ret;
}

}  // namespace devices
}  // namespace primitiv
