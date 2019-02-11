#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::max_fw_impl(const Tensor &x, std::uint32_t dim, Tensor &y) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t group_size = std::min(state_->max_fw_group_size, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      state_->max_fw_kernel[m].setArg(0, CDATA(x)); \
      state_->max_fw_kernel[m].setArg(1, s); \
      state_->max_fw_kernel[m].setArg(2, n); \
      state_->max_fw_kernel[m].setArg(3, MDATA(y)); \
      state_->queue.enqueueNDRangeKernel( \
          state_->max_fw_kernel[m], \
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
}

void OpenCL::max_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy,
    std::uint32_t dim, Tensor &gx) {
  const std::uint32_t n = x.shape()[dim];
  const std::uint32_t r = y.shape().size();
  const std::uint32_t s = y.shape().lower_volume(dim);
  std::uint32_t group_size = std::min(state_->max_bw_group_size, 1024u);
  while (group_size >> 1 >= n) group_size >>= 1;
  switch (group_size) {
#define CASE(k, m) \
    case k: \
      state_->max_bw_kernel[m].setArg(0, CDATA(x)); \
      state_->max_bw_kernel[m].setArg(1, CDATA(y)); \
      state_->max_bw_kernel[m].setArg(2, CDATA(gy)); \
      state_->max_bw_kernel[m].setArg(3, s); \
      state_->max_bw_kernel[m].setArg(4, n); \
      state_->max_bw_kernel[m].setArg(5, MDATA(gx)); \
      state_->queue.enqueueNDRangeKernel( \
          state_->max_bw_kernel[m], \
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
}

}  // namespace devices
}  // namespace primitiv
