/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <primitiv/opencl_device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/opencl_device.h>

using primitiv::devices::OpenCL;
using primitiv::c::internal::to_c_ptr;

extern "C" {

primitiv_Status primitiv_devices_OpenCL_new(
    uint32_t platform_id, uint32_t device_id, primitiv_Device **device) try {
  *device = to_c_ptr(new OpenCL(platform_id, device_id));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_devices_OpenCL_new_with_seed(
    uint32_t platform_id, uint32_t device_id, uint32_t rng_seed,
    primitiv_Device **device) try {
  *device = to_c_ptr(new OpenCL(platform_id, device_id, rng_seed));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_devices_OpenCL_num_platforms(
    uint32_t *num_platforms) try {
  *num_platforms = OpenCL::num_platforms();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_devices_OpenCL_num_devices(
    uint32_t platform_id, uint32_t *num_devices) try {
  *num_devices = OpenCL::num_devices(platform_id);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

}  // end extern "C"
