#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::max_pool2d_fw_impl(
    const Tensor &,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    Tensor &) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

void OpenCL::max_pool2d_bw_impl(
    const Tensor &, const Tensor &, const Tensor &,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    Tensor &) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

}  // namespace devices
}  // namespace primitiv
