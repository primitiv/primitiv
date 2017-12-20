/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <primitiv/opencl_device.h>

#include <primitiv/c/internal.h>
#include <primitiv/c/opencl_device.h>

using primitiv::devices::OpenCL;

#define CAST_TO_CC_OPENCL(x) reinterpret_cast<OpenCL*>(x)
#define CAST_TO_CONST_CC_OPENCL(x) reinterpret_cast<const OpenCL*>(x)

extern "C" {

primitiv_Device *primitiv_OpenCL_new(uint32_t platform_id, uint32_t device_id) {
  return to_c(new OpenCL(platform_id, device_id));
}
primitiv_Device *safe_primitiv_OpenCL_new(uint32_t platform_id,
                                          uint32_t device_id,
                                          primitiv_Status *status) {
  SAFE_RETURN(primitiv_OpenCL_new(platform_id, device_id), status, nullptr);
}

primitiv_Device *primitiv_OpenCL_new_with_seed(uint32_t platform_id,
                                               uint32_t device_id,
                                               uint32_t rng_seed) {
  return to_c(new OpenCL(platform_id, device_id, rng_seed));
}
primitiv_Device *safe_primitiv_OpenCL_new_with_seed(uint32_t platform_id,
                                                    uint32_t device_id,
                                                    uint32_t rng_seed,
                                                    primitiv_Status *status) {
  SAFE_RETURN(primitiv_OpenCL_new_with_seed(platform_id, device_id, rng_seed),
              status,
              nullptr);
}

void primitiv_OpenCL_delete(primitiv_Device *device) {
  delete CAST_TO_CC_OPENCL(device);
}
void safe_primitiv_OpenCL_delete(primitiv_Device *device,
                                 primitiv_Status *status) {
  SAFE_EXPR(primitiv_OpenCL_delete(device), status);
}

uint32_t primitiv_OpenCL_num_platforms() {
  return OpenCL::num_platforms();
}
uint32_t safe_primitiv_OpenCL_num_platforms(primitiv_Status *status) {
  SAFE_RETURN(primitiv_OpenCL_num_platforms(), status, 0);
}

uint32_t primitiv_OpenCL_num_devices(uint32_t platform_id) {
  return OpenCL::num_devices(platform_id);
}
uint32_t safe_primitiv_OpenCL_num_devices(uint32_t platform_id,
                                          primitiv_Status *status) {
  SAFE_RETURN(primitiv_OpenCL_num_devices(platform_id), status, 0);
}

void primitiv_OpenCL_dump_description(const primitiv_Device *device) {
  CAST_TO_CONST_CC_OPENCL(device)->dump_description();
}
void safe_primitiv_OpenCL_dump_description(const primitiv_Device *device,
                                         primitiv_Status *status) {
  SAFE_EXPR(primitiv_OpenCL_dump_description(device), status);
}

}  // end extern "C"
