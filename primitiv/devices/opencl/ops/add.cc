#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::add_const_fw_impl(const Tensor &x, float k, Tensor &y) {
  if (!state_->add_const_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->add_const_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/add.inc"
    }), "add_const_fw_kernel");
  }
  OPENCLDEV_FW_X_CONST(add_const);
}

void OpenCL::add_const_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) {
  if (!state_->add_const_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->add_const_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/add.inc"
    }), "add_const_bw_kernel");
  }
  OPENCLDEV_BW_X_CONST(add_const);
}

void OpenCL::add_scalar_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) {
  if (!state_->add_scalar_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->add_scalar_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/add.inc"
    }), "add_scalar_fw_kernel");
  }
  OPENCLDEV_FW_X_SCALAR(add_scalar);
}

void OpenCL::add_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
  if (!state_->add_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->add_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/add.inc"
    }), "add_fw_kernel");
  }
  OPENCLDEV_FW_AB(add)
}

void OpenCL::add_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  if (!state_->add_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->add_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/add.inc"
    }), "add_bw_kernel");
  }
  OPENCLDEV_BW_AB(add)
}

}  // namespace devices
}  // namespace primitiv
