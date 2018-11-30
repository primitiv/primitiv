#include <primitiv/config.h>

#include <primitiv/internal/opencl/utils.h>
#include <primitiv/devices/opencl/ops/common.h>

namespace primitiv {
namespace devices {

void OpenCL::multiply_const_fw_impl(const Tensor &x, float k, Tensor &y) {
  if (!state_->multiply_const_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->multiply_const_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/multiply.inc"
    }), "multiply_const_fw_kernel");
  }
  OPENCLDEV_FW_X_CONST(multiply_const);
}

void OpenCL::multiply_const_bw_impl(
    const Tensor &x, const Tensor &y, const Tensor &gy, float k, Tensor &gx) {
  if (!state_->multiply_const_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->multiply_const_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/multiply.inc"
    }), "multiply_const_bw_kernel");
  }
  OPENCLDEV_BW_X_CONST(multiply_const);
}

void OpenCL::multiply_scalar_fw_impl(const Tensor &x, const Tensor &k, Tensor &y) {
  if (!state_->multiply_scalar_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->multiply_scalar_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/multiply.inc"
    }), "multiply_scalar_fw_kernel");
  }
  OPENCLDEV_FW_X_SCALAR(multiply_scalar);
}

void OpenCL::multiply_fw_impl(const Tensor &a, const Tensor &b, Tensor &y) {
  if (!state_->multiply_fw_kernel.initialized()) {
    state_->initialize_kernel(state_->multiply_fw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/multiply.inc"
    }), "multiply_fw_kernel");
  }
  OPENCLDEV_FW_AB(multiply)
}

void OpenCL::multiply_bw_impl(
    const Tensor &a, const Tensor &b, const Tensor &y, const Tensor &gy,
    Tensor &ga, Tensor &gb) {
  if (!state_->multiply_bw_kernel.initialized()) {
    state_->initialize_kernel(state_->multiply_bw_kernel, std::string({
#include "primitiv/devices/opencl/kernels/common.inc"
#include "primitiv/devices/opencl/kernels/multiply.inc"
    }), "multiply_bw_kernel");
  }
  OPENCLDEV_BW_AB(multiply)
}

}  // namespace devices
}  // namespace primitiv
