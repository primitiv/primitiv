#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

#include <clblast.h>

namespace primitiv {
namespace devices {

void OpenCL::conv2d_fw_impl(const Tensor &, const Tensor &,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    Tensor &) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

void OpenCL::conv2d_bw_impl(
    const Tensor &, const Tensor &, const Tensor &, const Tensor &,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    std::uint32_t, std::uint32_t,
    Tensor &, Tensor &) {
  PRIMITIV_THROW_NOT_IMPLEMENTED;
}

}  // namespace devices
}  // namespace primitiv
