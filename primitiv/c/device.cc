/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <primitiv/device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/device.h>

using primitiv::Device;
using primitiv::c::internal::to_c_ptr;
using primitiv::c::internal::to_cpp_ptr;

extern "C" {

primitiv_Status primitiv_Device_get_default(primitiv_Device **device) try {
  *device = to_c_ptr(&Device::get_default());
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Device_set_default(primitiv_Device *device) try {
  PRIMITIV_C_CHECK_PTR_ARG(device);
  Device::set_default(*to_cpp_ptr(device));
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Device_delete(primitiv_Device *device) try {
  PRIMITIV_C_CHECK_PTR_ARG(device);
  delete to_cpp_ptr(device);
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

primitiv_Status primitiv_Device_dump_description(
    const primitiv_Device *device) try {
  PRIMITIV_C_CHECK_PTR_ARG(device);
  to_cpp_ptr(device)->dump_description();
  return ::primitiv_Status::PRIMITIV_OK;
} PRIMITIV_C_HANDLE_EXCEPTIONS

}  // end extern "C"
