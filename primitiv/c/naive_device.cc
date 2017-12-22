/* Copyright 2017 The primitiv Authors. All Rights Reserved. */
#include <primitiv/config.h>

#include <primitiv/naive_device.h>
#include <primitiv/c/internal.h>
#include <primitiv/c/naive_device.h>

using primitiv::devices::Naive;
using primitiv::c::internal::to_c_ptr;

extern "C" {

primitiv_Status primitiv_devices_Naive_new(primitiv_Device **device) try {
  *device = to_c_ptr(new Naive());
  return ::primitiv_Status::PRIMITIV_OK;
} HANDLE_EXCEPTION

primitiv_Status primitiv_devices_Naive_new_with_seed(
    uint32_t seed, primitiv_Device **device) try {
  *device = to_c_ptr(new Naive(seed));
  return ::primitiv_Status::PRIMITIV_OK;
} HANDLE_EXCEPTION

}  // end extern "C"
