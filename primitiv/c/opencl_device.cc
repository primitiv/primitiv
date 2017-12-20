/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <config.h>

#include <primitiv/opencl_device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/opencl_device.h>

using primitiv::devices::OpenCL;
using primitiv::c::internal::to_c_ptr;

#define CAST_TO_CC_OPENCL(x) reinterpret_cast<OpenCL*>(x)
#define CAST_TO_CONST_CC_OPENCL(x) reinterpret_cast<const OpenCL*>(x)

extern "C" {

primitiv_Status primitiv_devices_OpenCL_new(primitiv_Device **device,
                                    uint32_t platform_id, uint32_t device_id) {
  try {
    *device = to_c_ptr(new OpenCL(platform_id, device_id));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

primitiv_Status primitiv_devices_OpenCL_new_with_seed(
    primitiv_Device **device, uint32_t platform_id, uint32_t device_id,
    uint32_t rng_seed) {
  try {
    *device = to_c_ptr(new OpenCL(platform_id, device_id, rng_seed));
    return ::primitiv_Status::PRIMITIV_OK;
  } HANDLE_EXCEPTION
}

void primitiv_devices_OpenCL_delete(primitiv_Device *device) {
  delete CAST_TO_CC_OPENCL(device);
}

uint32_t primitiv_devices_OpenCL_num_platforms() {
  return OpenCL::num_platforms();
}

uint32_t primitiv_devices_OpenCL_num_devices(uint32_t platform_id) {
  return OpenCL::num_devices(platform_id);
}

void primitiv_devices_OpenCL_dump_description(const primitiv_Device *device) {
  CAST_TO_CONST_CC_OPENCL(device)->dump_description();
}

}  // end extern "C"

#undef CAST_TO_CC_OPENCL
#undef CAST_TO_CONST_CC_OPENCL
