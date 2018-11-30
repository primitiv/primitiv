#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::subtract_const_r_fw_impl(const Tensor &x, float k, Tensor &y) {
  if (!state_->subtract_const_r_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->subtract_const_r_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/subtract.inc"
    }), "subtract_const_r_fw_kernel");
  }
  OPENCLDEV_FW_X_CONST(subtract_const_r);
}

void OpenCL::subtract_const_l_fw_impl(const Tensor &x, float k, Tensor &y) {
  if (!state_->subtract_const_l_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->subtract_const_l_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/subtract.inc"
    }), "subtract_const_l_fw_kernel");
  }
  OPENCLDEV_FW_X_CONST(subtract_const_l);
}

void OpenCL::subtract_const_r_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) {
  if (!state_->subtract_const_r_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->subtract_const_r_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/subtract.inc"
    }), "subtract_const_r_bw_kernel");
  }
  OPENCLDEV_BW_X_CONST(subtract_const_r);
}

void OpenCL::subtract_const_l_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) {
  if (!state_->subtract_const_l_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->subtract_const_l_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/subtract.inc"
    }), "subtract_const_l_bw_kernel");
  }
  OPENCLDEV_BW_X_CONST(subtract_const_l);
}

void OpenCL::subtract_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) {
  if (!state_->subtract_scalar_r_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->subtract_scalar_r_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/subtract.inc"
    }), "subtract_scalar_r_fw_kernel");
  }
  OPENCLDEV_FW_X_SCALAR(subtract_scalar_r);
}

void OpenCL::subtract_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) {
  if (!state_->subtract_scalar_l_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->subtract_scalar_l_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/subtract.inc"
    }), "subtract_scalar_l_fw_kernel");
  }
  OPENCLDEV_FW_X_SCALAR(subtract_scalar_l);
}

void OpenCL::subtract_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
  if (!state_->subtract_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->subtract_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/subtract.inc"
    }), "subtract_fw_kernel");
  }
  OPENCLDEV_FW_AB(subtract);
}

void OpenCL::subtract_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  if (!state_->subtract_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->subtract_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/subtract.inc"
    }), "subtract_bw_kernel");
  }
  OPENCLDEV_BW_AB(subtract);
}

}  // namespace devices
}  // namespace primitiv
