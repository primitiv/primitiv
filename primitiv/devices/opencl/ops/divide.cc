#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::divide_const_r_fw_impl(const Tensor &x, float k, Tensor &y) {
  if (!state_->divide_const_r_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->divide_const_r_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/divide.inc"
    }), "divide_const_r_fw_kernel");
  }
  OPENCLDEV_FW_X_CONST(divide_const_r);
}

void OpenCL::divide_const_l_fw_impl(const Tensor &x, float k, Tensor &y) {
  if (!state_->divide_const_l_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->divide_const_l_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/divide.inc"
    }), "divide_const_l_fw_kernel");
  }
  OPENCLDEV_FW_X_CONST(divide_const_l);
}

void OpenCL::divide_const_r_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) {
  if (!state_->divide_const_r_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->divide_const_r_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/divide.inc"
    }), "divide_const_r_bw_kernel");
  }
  OPENCLDEV_BW_X_CONST(divide_const_r);
}

void OpenCL::divide_const_l_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) {
  if (!state_->divide_const_l_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->divide_const_l_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/divide.inc"
    }), "divide_const_l_bw_kernel");
  }
  OPENCLDEV_BW_X_CONST(divide_const_l);
}

void OpenCL::divide_scalar_r_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) {
  if (!state_->divide_scalar_r_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->divide_scalar_r_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/divide.inc"
    }), "divide_scalar_r_fw_kernel");
  }
  OPENCLDEV_FW_X_SCALAR(divide_scalar_r);
}

void OpenCL::divide_scalar_l_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) {
  if (!state_->divide_scalar_l_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->divide_scalar_l_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/divide.inc"
    }), "divide_scalar_l_fw_kernel");
  }
  OPENCLDEV_FW_X_SCALAR(divide_scalar_l);
}

void OpenCL::divide_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
  if (!state_->divide_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->divide_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/divide.inc"
    }), "divide_fw_kernel");
  }
  OPENCLDEV_FW_AB(divide);
}

void OpenCL::divide_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  if (!state_->divide_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->divide_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/divide.inc"
    }), "divide_bw_kernel");
  }
  OPENCLDEV_BW_AB(divide);
}

}  // namespace devices
}  // namespace primitiv
